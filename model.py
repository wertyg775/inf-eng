from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

@torch.inference_mode()
def generate_with_cache(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated = input_ids
    past_key_values=None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=1, keepdim=True)

        past_key_values = outputs.past_key_values
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = next_token
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )

    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        



if __name__ == "__main__":
    download_model()
