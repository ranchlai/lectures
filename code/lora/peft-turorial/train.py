# -*- coding: utf-8 -*-
import argparse

import numpy as np
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def get_args():
    # add arguments, with_lora
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_lora", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    if args.with_lora:
        model_id = "google/flan-t5-large"  # or large
        peft_model_id = "results_large"  # or large
        print("using lora for training large model")
    else:

        model_id = "google/flan-t5-base"  # or large
        peft_model_id = "results_base"  # or large
        print("not using lora for training base model")

    # Load dataset from the hub
    dataset = load_dataset("samsum")
    # use only a portion of the dataset
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    dataset["train"] = dataset["train"].select(range(0, 256))
    dataset["test"] = dataset["test"].select(range(0, 32))
    # Train dataset size: 14732
    # Test dataset size: 819
    print("using only a portion of the dataset")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["dialogue"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
    )
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    max_source_length = int(np.percentile(input_lenghts, 85))
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["summary"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
    )
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, 90))
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=sample["summary"],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"]
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # save datasets to disk for later easy loading
    tokenized_dataset["train"].save_to_disk("data/train")
    tokenized_dataset["test"].save_to_disk("data/eval")

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    # prepare int-8 model for training
    # model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    if args.with_lora:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    output_dir = peft_model_id

    # trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        # batch_size=8,
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=3e-4,  # higher learning rate
        num_train_epochs=10,
        # per_device_train_batch_size=8,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard",
        lr_scheduler_type="constant",
        # using only a portion of the dataset
        # train = 30,
        # max_eval_samples = 30,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    # train model
    trainer.train()

    # Save our LoRA model & tokenizer results
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    # if you want to save the base model to call
    # trainer.model.base_model.save_pretrained(peft_model_id)
