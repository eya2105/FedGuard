from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01_preprocessing_dataset.ipynb": "scripts/01_preprocessing_dataset.py",
    "02_benchmark_models.ipynb": "scripts/02_benchmark_models.py",
    "03_autoencoder_failure_inspection.ipynb": "scripts/03_autoencoder_failure_inspection.py",
    "04_centralized_model.ipynb": "scripts/04_centralized_model.py",
    "05_federated_learning_implementation.ipynb": "scripts/05_federated_learning_implementation.py",
    "06_final_hil.ipynb": "scripts/06_final_hil.py",
}

def load_ipynb(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_code_cells(nb_json):
    cells = nb_json.get("cells", [])
    code_blocks = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            code_blocks.append(src)
    return code_blocks

def export_notebook(notebook_path: Path, script_path: Path):
    nb = load_ipynb(notebook_path)
    code_blocks = get_code_cells(nb)

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Auto-exported from {notebook_path.name}\n")
        for i, block in enumerate(code_blocks):
            f.write(f"# %% [cell {i+1}]\n")
            f.write(block)
            if not block.endswith("\n"):
                f.write("\n")
            f.write("\n")

def main():
    for nb_name, py_rel in NOTEBOOKS.items():
        nb_path = ROOT / nb_name
        py_path = ROOT / py_rel
        if not nb_path.exists():
            print(f"SKIP (not found): {nb_path}")
            continue
        export_notebook(nb_path, py_path)
        print(f"OK: {nb_path.name} -> {py_rel}")

if __name__ == "__main__":
    main()