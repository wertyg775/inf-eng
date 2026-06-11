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

def tokenize_text(model, tokenizer, text):
    return tokenizer(text, return_tensors="pt").to(model.device)

@torch.inference_mode()
def generate_with_cache(model, tokenizer, input_ids, attention_mask, past_key_values=None, max_new_tokens=128):

    new_tokens = []

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
        new_tokens.append(next_token)

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

    generated_ids = torch.cat(new_tokens, dim=1)

    return {
        "text": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        "generated_ids": generated_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values
    }
        



if __name__ == "__main__":
    download_model()
