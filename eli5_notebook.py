
# https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/language_modeling.ipynb
# the same:
# https://github.com/diixo/notebooks/blob/main/transformers_doc/en/language_modeling.ipynb

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from modeling_gptr import GPTRForCausalLM


block_size = 512
BATCH_SZ = 7
EPOCHS = 1

NUM_PROC = 4


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("GPU available for training:", torch.cuda.device_count(), "device(s)")
        try:
            print("Current CUDA device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            pass
    else:
        print("CUDA not available, training will run on CPU.")

    model = GPTRForCausalLM.from_pretrained("aitetic/gpt-r-0.3b", map_location=device)

    tokenizer = GPT2TokenizerFast.from_pretrained("aitetic/gpt-r-0.3b")

    ###################################################################

    #sym:cache_dir — use project-local Hugging Face cache directory for datasets
    cache_dir = os.path.abspath("./.hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    eli5 = load_dataset("aitetic/eli5-lfqa-combined", split="train")

    # The local JSON has one field 'text' per record. Tokenize directly from that field.
    def preprocess_function(examples):
        # examples['text'] is a list of strings when batched
        return tokenizer(examples["text"])


    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=eli5.column_names,
        batch_size=1000,
    )


    lm_dataset = tokenized_eli5.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        batch_size=1000,
    )

    ##########################################################

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ##########################################################

    output_dir = "./train_products"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        learning_rate=8e-5,
        num_train_epochs=EPOCHS,
        weight_decay=0.0,
        save_total_limit=1,

        save_strategy="steps",  # "epoch"
        save_steps=10_000,

        save_safetensors=False,
        push_to_hub=False,
        per_device_train_batch_size=BATCH_SZ,
        lr_scheduler_type="constant",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Trainer device:", trainer.args.device)

    checkpoint = get_last_checkpoint(training_args.output_dir)
    if checkpoint is not None:
        print(f"Resuming training from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()


    # The Trainer has already saved checkpoint folders automatically during training
    # in the output_dir according to save_strategy/save_steps.
    # Those folders include optimizer, scheduler, model weights, and trainer state.
    print(f"Checkpoint folders are saved automatically to {training_args.output_dir}")

    # Also save the final model weights in PyTorch .pt format.
    # This is useful if you want a standalone weights file, but it does not replace the Trainer checkpoint.
    # Pass training metadata via train_config to keep hyperparameters with the saved model.
    model.save_model(output_dir, train_config=training_args.to_dict())

    # To continue training later from the Trainer checkpoint, keep the output_dir folder intact
    # and call trainer.train(resume_from_checkpoint=last_checkpoint) on the same or a new Trainer.
    # For example:
    #   from transformers.trainer_utils import get_last_checkpoint
    #   last_checkpoint = get_last_checkpoint(output_dir)
    #   trainer.train(resume_from_checkpoint=last_checkpoint)

    # To continue training later from the saved checkpoint folder, you can load the model back and create a new Trainer:
    #   model = GPTRForCausalLM.from_pretrained(output_dir + "/pt_model/model.pt")
    #   trainer = Trainer(model=model, args=training_args, train_dataset=lm_dataset, data_collator=data_collator, processing_class=tokenizer)
    #   trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    main()
