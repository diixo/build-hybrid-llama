
import json
from auto_config import AutoConfigModel
import torch
from transformers import GPT2TokenizerFast


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained(f"aitetic/gpt-r-0.3b", local_files_only=False)

    model = AutoConfigModel.from_pretrained("aitetic/gpt-r-0.3b", map_location=device)

    if model is None:
        raise SystemExit("Checkpoint 'aitetic/gpt-r-0.3b' was not found on Hugging Face Hub or in the local cache.")

    config_obj = getattr(model, "config", None)
    config_dict = vars(config_obj) if config_obj is not None else {}

    print("Model configuration:")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False, default=str))

    model = model.to(device)
    
    input_ids = tokenizer("Neural Network", return_tensors="pt").input_ids.to(device)
    text = model.generate(input_ids, max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)

    gen_ids = text[0].detach().cpu().tolist()

    print(f"Generated text: {tokenizer.decode(gen_ids, skip_special_tokens=True)}")
