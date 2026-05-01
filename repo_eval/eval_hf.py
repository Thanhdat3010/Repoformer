#!/usr/bin/env python
# coding=utf-8

import os
import json
import torch
import torch.nn.functional as F
import logging
import argparse
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from collections import Counter

from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    default_data_collator,
    AutoTokenizer,
    set_seed,
    AutoModelForCausalLM
)

from eval_metric import compute_metric_stmt
from eval_metric_cceval import compute_metric_stmt_cceval
from datetime import datetime
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_fim_tokens(model_name):
    """Return the correct FIM special tokens based on model family."""
    name = model_name.lower()
    if 'qwen' in name:
        return '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>'
    if 'deepseek' in name:
        return '<｜fim begin｜>', '<｜fim end｜>', '<｜fim hole｜>'
    # StarCoder / default
    return '<fim_prefix>', '<fim_suffix>', '<fim_middle>'


def custom_data_collator(features):
    from torch.nn.utils.rnn import pad_sequence
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if k in ["input_ids", "attention_mask", "labels"]:
                # Pad sequences to the same length
                sequences = [torch.tensor(f[k]) if not isinstance(f[k], torch.Tensor) else f[k] for f in features]
                batch[k] = pad_sequence(sequences, batch_first=True, padding_value=0)
            elif isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if v is not None and isinstance(v, str):
            batch[k] = [f[k] for f in features]

    return batch


def build_datasets(args, tokenizer):
    # Initialize the model and tokenizer
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token

    # load the files into Dataset
    raw_datasets = load_dataset("json", data_files=args.prompt_file, cache_dir=args.cache_dir)
    raw_datasets = raw_datasets["train"]
    raw_datasets = raw_datasets.map(lambda example, idx: {'index': idx, **example}, with_indices=True)
    index2taskid = {idx: md["task_id"] for idx, md in zip(raw_datasets["index"], raw_datasets["metadata"])}
    column_names = raw_datasets.column_names

    def prepare_features(examples):
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length - args.gen_length
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_fim(examples):
        fim_prefix, fim_suffix, fim_middle = get_fim_tokens(args.model_name_or_path)
        # first do proper truncation 
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            examples["prompt"],
            padding=False,
            max_length=args.max_seq_length - args.gen_length - 10,
            truncation=True,
        )
        # inject fim tokens and redo tokenization
        if 'deepseek' in args.model_name_or_path.lower():
            # DeepSeek: <begin>PRE<hole>SUF<end>
            input_text = [fim_prefix + x + fim_middle + "" + fim_suffix for x in tokenizer.batch_decode(tokenized_inputs['input_ids'])]
        else:
            # StarCoder/Qwen: <pre>PRE<suf>SUF<mid>
            input_text = [fim_prefix + x + fim_suffix + fim_middle for x in tokenizer.batch_decode(tokenized_inputs['input_ids'])]
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            # truncation=True,
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_cfc_fim(examples):
        fim_prefix, fim_suffix, fim_middle = get_fim_tokens(args.model_name_or_path)
        in_file_seq_length = args.max_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        cfc_features = tokenizer(
            examples["crossfile_context"] if isinstance(examples["crossfile_context"], str) else [x['text'] if isinstance(x, dict) else x for x in examples["crossfile_context"]],
            padding=False,
            truncation=True,
            max_length=args.cfc_seq_length - 5
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length - 5
        )

        if 'deepseek' in args.model_name_or_path.lower():
            # DeepSeek: <begin>PRE<hole>SUF<end>
            input_text = [fim_prefix + y + fim_middle + x + fim_suffix for x, y in zip(tokenizer.batch_decode(cfc_features['input_ids']),
                                                                                       tokenizer.batch_decode(infile_seq_features['input_ids']))]
        else:
            # StarCoder/Qwen: <pre>PRE<suf>SUF<mid>
            input_text = [fim_prefix + y + fim_suffix + x + fim_middle for x, y in zip(tokenizer.batch_decode(cfc_features['input_ids']),
                                                                                       tokenizer.batch_decode(infile_seq_features['input_ids']))]
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            truncation=False
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_cfc(examples):
        in_file_seq_length = args.max_seq_length - args.cfc_seq_length - args.gen_length

        tokenizer.truncation_side = "right"
        crossfile_seq_features = tokenizer(
            examples["crossfile_context"] if isinstance(examples["crossfile_context"], str) else [x['text'] if isinstance(x, dict) else x for x in examples["crossfile_context"]],
            truncation=True,
            max_length=args.cfc_seq_length
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length
        )

        # concatenate project-level context and file-level context
        features = {}
        for k, v in infile_seq_features.items():
            features[k] = []
            for idx, e in enumerate(v):
                iids = crossfile_seq_features[k][idx] + e
                features[k].append(iids)

        # pad to max_seq_length
        input_ids = features["input_ids"]
        for idx, iids in enumerate(input_ids):
            if len(iids) < args.max_seq_length - args.gen_length:
                input_ids[idx] = [tokenizer.pad_token_id] * (args.max_seq_length - args.gen_length - len(iids)) + iids
            else:
                input_ids[idx] = iids[:args.max_seq_length - args.gen_length]
        
        attention_mask = [[0 if iid == tokenizer.pad_token_id else 1 for iid in iids] for iids in input_ids]
        features["input_ids"] = input_ids
        features["attention_mask"] = attention_mask
        features["index"] = examples["index"]
        return features

    def prepare_features_right_context_fim(examples):
        fim_prefix, fim_suffix, fim_middle = get_fim_tokens(args.model_name_or_path)
        in_file_seq_length = args.max_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        right_context_features = tokenizer(
            examples["right_context"],
            padding=False,
            truncation=True,
            max_length=args.right_context_length - 5
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length - 5
        )

        if 'deepseek' in args.model_name_or_path.lower():
            # DeepSeek: <begin>PRE<hole>SUF<end>
            input_text = [fim_prefix + y + fim_middle + x + fim_suffix for x, y in zip(tokenizer.batch_decode(right_context_features['input_ids']),
                                                                                       tokenizer.batch_decode(infile_seq_features['input_ids']))]
        else:
            # StarCoder: <pre>PRE<suf>SUF<mid>
            input_text = [fim_prefix + y + fim_suffix + x + fim_middle for x, y in zip(tokenizer.batch_decode(right_context_features['input_ids']),
                                                                                       tokenizer.batch_decode(infile_seq_features['input_ids']))]
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            truncation=False
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features

    def prepare_features_right_cfc_left_fim(examples):
        fim_prefix, fim_suffix, fim_middle = get_fim_tokens(args.model_name_or_path)
        in_file_seq_length = args.max_seq_length - args.right_context_length - args.cfc_seq_length - args.gen_length

        tokenizer.truncation_side = "right"
        right_context_features = tokenizer(
            examples["right_context"],
            padding=False,
            truncation=True,
            max_length=args.right_context_length - 5
        )
        tokenizer.truncation_side = "right"
        cfc_features = tokenizer(
            examples["crossfile_context"] if isinstance(examples["crossfile_context"], str) else [x['text'] if isinstance(x, dict) else x for x in examples["crossfile_context"]],
            padding=False,
            truncation=True,
            max_length=args.cfc_seq_length - 5
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length - 5
        )

        if 'deepseek' in args.model_name_or_path.lower():
            # DeepSeek: <begin>PRE<hole>SUF<end>
            input_text = [fim_prefix + y + fim_middle + z + x + fim_suffix for x, y, z in zip(tokenizer.batch_decode(right_context_features['input_ids']),
                                                                                              tokenizer.batch_decode(infile_seq_features['input_ids']),
                                                                                              tokenizer.batch_decode(cfc_features['input_ids']))]
        else:
            # StarCoder: <pre>PRE<suf>SUF<mid>
            input_text = [fim_prefix + y + fim_suffix + z + x + fim_middle for x, y, z in zip(tokenizer.batch_decode(right_context_features['input_ids']),
                                                                                              tokenizer.batch_decode(infile_seq_features['input_ids']),
                                                                                              tokenizer.batch_decode(cfc_features['input_ids']))]
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            truncation=False
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features

    if args.model_type == "codelm":
        if args.use_fim_prompt:
            tokenized_datasets = raw_datasets.map(
                prepare_features_fim,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                prepare_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif args.model_type == "codelm_cfc":
        if args.use_fim_prompt:
            tokenized_datasets = raw_datasets.map(
                prepare_features_cfc_fim,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                prepare_features_cfc,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif args.model_type == "codelm_leftright_context":
        tokenized_datasets = raw_datasets.map(
            prepare_features_right_context_fim,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    elif args.model_type == "codelm_right_cfc_left":
        tokenized_datasets = raw_datasets.map(
            prepare_features_right_cfc_left_fim,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
    return tokenized_datasets, index2taskid


def model_inference(tokenized_datasets, index2taskid, tokenizer):
    dataloader = DataLoader(
        tokenized_datasets, 
        sampler=SequentialSampler(tokenized_datasets), 
        batch_size=args.batch_size, 
        collate_fn=custom_data_collator
    )
    
    load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": args.cache_dir,
        "device_map": "auto",
    }
    
    if args.dtype == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "int8":
        load_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    logger.info(f"model.dtype={model.dtype}")
    logger.info(args)
    logger.info(f"total samples: {len(tokenized_datasets)}")

    model, dataloader = accelerator.prepare(model, dataloader)
    
    # generate
    results = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            gen_kwargs = {
                "max_new_tokens": args.gen_length,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_return_sequences": args.num_return_sequences,
                "repetition_penalty": args.repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if args.stop_token:
                gen_kwargs["stopping_criteria"] = [args.stop_token]
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
            
            # Decode only the generated part
            gen_ids = outputs[:, input_ids.shape[-1]:]
            decoded_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            for i, output in enumerate(decoded_outputs):
                # Manually strip any leaked FIM tokens from DeepSeek
                output = output.replace('<｜fim begin｜>', '').replace('<｜fim hole｜>', '').replace('<｜fim end｜>', '')
                output = output.replace('<\uff5cfim begin\uff5c>', '').replace('<\uff5cfim hole\uff5c>', '').replace('<\uff5cfim end\uff5c>', '')
                
                sample_idx = batch["index"][i].item()
                task_id = index2taskid[sample_idx]
                results.append({
                    "task_id": task_id,
                    "pred": output
                })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/prediction.jsonl", 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument("--model_name_or_path", type=str, default="bigcode/starcoderbase")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="codelm", 
        choices=["codelm", "codelm_cfc", "codelm_leftright_context", "codelm_right_cfc_left"]
    )
    parser.add_argument("--use_fim_prompt", action='store_true', help="use fim prompt")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument("--gen_length", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--cfc_seq_length", type=int, default=512)
    parser.add_argument("--right_context_length", type=int, default=512)
    parser.add_argument("--min_cfc_score", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "int8"])
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--drop_outliner_lengths", action='store_true')
    parser.add_argument("--ts_lib", type=str, default=None)
    parser.add_argument("--only_compute_metric", action='store_true')
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["line_completion", "api_completion", "function_completion"],
        default="line_completion",
        help="task name"
    )
    parser.add_argument("--draft_model", type=str, default=None, help="draft model for speculative decoding")
    parser.add_argument("--lookahead", type=int, default=None, help="lookahead for speculative decoding")
    parser.add_argument("--lookahead_strategy", type=str, default=None, choices=['heuristic', 'constant'],
                        help="strategy for setting the lookahead for speculative decoding")
    parser.add_argument("--compute_cceval_metric", action='store_true', help="use cceval metric")
    parser.add_argument("--log_latency", action='store_true', help="log latency in the results file")
    parser.add_argument("--log_uncertainty", action='store_true', help="log uncertainty in the results file")
    
    args = parser.parse_args()

    accelerator = Accelerator()
    if not args.only_compute_metric:
        if args.tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
        else:
            if 'no_fim' in args.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path.strip('no_fim'), trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        
        set_seed(args.seed)
        tokenized_datasets, index2taskid = build_datasets(args, tokenizer)

        # Measure prompt generation time
        import time as _time
        _prompt_gen_start = _time.time()
        # build_datasets is already called above, we just measure a second pass or reuse
        _prompt_gen_end = _time.time()
        
        model_inference(tokenized_datasets, index2taskid, tokenizer)

    # Metric computation phase
    if accelerator.is_main_process:
        if args.compute_cceval_metric:
            compute_metric_stmt_cceval(args)
        else:
            compute_metric_stmt(args)
            
        # PRINT FINAL METRICS SUMMARY BOX
        try:
            import json as _json
            result_file = os.path.join(args.output_dir, "results.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as _f:
                    _res = _json.load(_f)
                    print("\n" + "#"*60)
                    print(f" EVALUATION COMPLETE: {args.task.upper()}")
                    print(f" MODEL:  {args.model_name_or_path}")
                    print("-" * 60)
                    print(f" Exact Match (EM):      {_res.get('em', 'N/A')}%")
                    print(f" Edit Similarity (ES):  {_res.get('es', 'N/A')}")
                    print(f" Samples Evaluated:     {_res.get('total', 'N/A')}")
                    print("#"*60 + "\n")
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
