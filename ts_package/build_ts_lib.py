"""Build tree-sitter language libraries.

Compatible with both tree-sitter < 0.22 (build_library API)
and tree-sitter >= 0.22 (pip-installed language packages).
"""
import importlib
import sys

# Languages needed for evaluation
LANGUAGES = {
    "python": "tree_sitter_python",
    "java": "tree_sitter_java",
    "typescript": "tree_sitter_typescript",
    "csharp": "tree_sitter_c_sharp",
}


def check_new_api():
    """Check if we're using tree-sitter >= 0.22 (new API)."""
    from tree_sitter import Language
    return not hasattr(Language, 'build_library')


def build_with_old_api():
    """Build .so files using tree-sitter < 0.22 API."""
    from tree_sitter import Language
    for lang in ["java", "python", "typescript", "csharp"]:
        ts_lang = "c-sharp" if lang == "csharp" else lang
        if lang == "typescript":
            git_dir = f"ts_package/tree-sitter-{ts_lang}/{lang}"
        else:
            git_dir = f"ts_package/tree-sitter-{ts_lang}"
        Language.build_library(f'build/{lang}-lang-parser.so', [git_dir])
    print("Built .so files in build/ directory.")


def install_language_packages():
    """Install tree-sitter language packages via pip for >= 0.22."""
    import subprocess
    packages = [
        "tree-sitter-python",
        "tree-sitter-java",
        "tree-sitter-typescript",
        "tree-sitter-c-sharp",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("Installed tree-sitter language packages.")


def get_language(lang_name):
    """Get a Language object for the given language name.
    
    Works with both old API (.so files) and new API (pip packages).
    """
    from tree_sitter import Language
    
    if check_new_api():
        # tree-sitter >= 0.22: use pip-installed packages
        module_name = LANGUAGES.get(lang_name)
        if module_name is None:
            raise ValueError(f"Unknown language: {lang_name}")
        mod = importlib.import_module(module_name)
        return Language(mod.language())
    else:
        # tree-sitter < 0.22: use .so files
        ts_lang = "c_sharp" if lang_name == "csharp" else lang_name
        so_path = f"build/{lang_name}-lang-parser.so"
        return Language(so_path, ts_lang)


if __name__ == "__main__":
    if check_new_api():
        print("tree-sitter >= 0.22 detected. Installing language packages...")
        install_language_packages()
        # Verify
        for lang in LANGUAGES:
            try:
                l = get_language(lang)
                print(f"  ✓ {lang}")
            except Exception as e:
                print(f"  ✗ {lang}: {e}")
    else:
        print("tree-sitter < 0.22 detected. Building .so files...")
        build_with_old_api()
