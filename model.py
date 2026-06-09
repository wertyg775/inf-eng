from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
if Path.cwd().parent.name == "proj1":
    root_dir = Path.cwd().parent
else:
    root_dir = Path.cwd()
'''
root_dir = Path(__file__).resolve().parent
LOCAL_DIR = root_dir / "weights" / "qwen2.5-1.5B"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

def download_model(model_name: str=MODEL_NAME, saved_dir: str=LOCAL_DIR):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    saved_dir = Path(saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(saved_dir)
    tokenizer.save_pretrained(saved_dir)

def load_model(saved_dir: str=LOCAL_DIR):
    saved_dir = Path(saved_dir)
    if not saved_dir.exists():
        raise FileNotFoundError(
            f"No local weights found at {saved_dir}. Run download_model() first"
        )
    model = AutoModelForCausalLM.from_pretrained(saved_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_dir)

    return model, tokenizer


if __name__ == "__main__":
    download_model()
