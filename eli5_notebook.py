
# https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/language_modeling.ipynb

import os

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from model_llama import GPTRForCausalLM

block_size = 256


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

    tokenizer = GPT2TokenizerFast.from_pretrained("aitetic/gpt-r-0.3b")

    model = GPTRForCausalLM.from_pretrained("aitetic/gpt-r-0.3b")

    # Simple text generation example after loading the model and tokenizer
    model.eval()
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("\n=== Generation example ===")
    print("Prompt:", prompt)
    print("Generated:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    print("=== End generation ===\n")

    ###################################################################

    eli5 = load_dataset("dany0407/eli5_category", split="train")

    eli5 = eli5.select(range(1000))

    eli5 = eli5.flatten()

    #print(eli5[0])


    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])


    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5.column_names,
        batch_size=1000,
    )


    lm_dataset = tokenized_eli5.map(
        group_texts,
        batched=True,
        num_proc=4,
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
        num_train_epochs=1,
        weight_decay=0.0,
        save_strategy="epoch",
        #save_steps=500,
        #save_total_limit=1,
        save_safetensors=False,
        push_to_hub=False,
        per_device_train_batch_size=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # checkpoint = get_last_checkpoint(training_args.output_dir)
    # if checkpoint is not None:
    #     print(f"Resuming training from checkpoint: {checkpoint}")
    #     trainer.train(resume_from_checkpoint=checkpoint)
    # else:
    #     trainer.train()


    # The Trainer has already saved checkpoint folders automatically during training
    # in the output_dir according to save_strategy/save_steps.
    # Those folders include optimizer, scheduler, model weights, and trainer state.
    print(f"Checkpoint folders are saved automatically to {training_args.output_dir}")

    # Also save the final model weights in PyTorch .pt format.
    # This is useful if you want a standalone weights file, but it does not replace the Trainer checkpoint.
    # Pass training metadata via train_config to keep hyperparameters with the saved model.
    model.save_model(output_dir + "/pt_model", train_config=training_args.to_dict())

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
