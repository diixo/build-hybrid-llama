
import json
import torch
from transformers import GPT2TokenizerFast
from modeling_gptr import GPTRForCausalLM


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = GPTRForCausalLM.from_pretrained("aitetic/gptr-noomo-0.3b", map_location=device)

    if model is None:
        raise SystemExit("Checkpoint 'aitetic/gptr-noomo-0.3b' was not found on Hugging Face Hub or in the local cache.")

    config_obj = getattr(model, "config", None)
    config_dict = vars(config_obj) if config_obj is not None else {}

    print("Model configuration:")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False, default=str))

    model = model.to(device)
    ############################################################################

    tokenizer = GPT2TokenizerFast.from_pretrained(f"aitetic/gptr-noomo-0.3b", local_files_only=False)

    prompt = "The future of AI is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    output_ids = generated_ids[0].detach().cpu().tolist()

    print("\n=== Generation example ===")
    print("Prompt:", prompt)
    print("Generated:", tokenizer.decode(output_ids, skip_special_tokens=True))
    print("=== End generation ===\n")

