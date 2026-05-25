
from auto_config import AutoConfigModel
import torch
from transformers import GPT2TokenizerFast


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained(f"data/gpt-noomo-32k", local_files_only=True)

    model = AutoConfigModel.from_pretrained("aitetic/gpt-r-0.3b-base", map_location=device)

    if model is None:
        raise SystemExit("Checkpoint 'aitetic/gpt-r-0.3b-base' was not found on Hugging Face Hub or in the local cache.")

    model = model.to(device)
    
    input_ids = tokenizer("I am Language Model", return_tensors="pt").input_ids.to(device)
    text = model.generate(input_ids, max_new_tokens=80)

    print("Generated text:", tokenizer.decode(text[0].detach().cpu().tolist(), skip_special_tokens=True))
