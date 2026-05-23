
from auto_config import AutoConfigModel
from transformers import GPT2TokenizerFast


if __name__ == "__main__":

    tokenizer = GPT2TokenizerFast.from_pretrained(f"data/gpt-noomo-32k", local_files_only=True)

    model = AutoConfigModel.from_pretrained("aitetic/gpt-r-0.3b-warmup")

    if model is None:
        raise SystemExit("Checkpoint 'aitetic/gpt-r-0.3b-warmup' was not found on Hugging Face Hub or in the local cache.")
    
    input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids
    text = model.generate(input_ids, max_new_tokens=80)

    print("Generated text:", tokenizer.decode(text[0], skip_special_tokens=True))
