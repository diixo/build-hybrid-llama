
# https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/language_modeling.ipynb


from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer


def main():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    eli5 = load_dataset("dany0407/eli5_category", split="train")


    eli5 = eli5.flatten()

    print(eli5[0])


    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])


    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5.column_names,
        batch_size=1000,
    )

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

    lm_dataset = tokenized_eli5.map(
        group_texts,
        batched=True,
        num_proc=4,
        batch_size=1000,
    )

    exit(0)
    ##########################################################

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ##########################################################

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir="my_awesome_eli5_clm-model",
        eval_strategy="no",
        learning_rate=8e-5,
        num_train_epochs=1,
        weight_decay=0.0,
        push_to_hub=False,
        per_device_train_batch_size = 4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    #trainer.train()


if __name__ == "__main__":
    main()
