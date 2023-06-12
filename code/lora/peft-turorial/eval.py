# -*- coding: utf-8 -*-
import argparse
import random
from random import randrange

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_args():
    # add arguments, with_lora
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_lora", action="store_true")
    return parser.parse_args()


def evaluate_peft_model(sample, max_target_length=50):
    # generate summary
    outputs = model.generate(
        input_ids=sample["input_ids"].unsqueeze(0).cuda(),
        do_sample=True,
        top_p=0.9,
        max_new_tokens=max_target_length,
    )
    prediction = tokenizer.decode(
        outputs[0].detach().cpu().numpy(), skip_special_tokens=True
    )
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(
        sample["labels"] != -100, sample["labels"], tokenizer.pad_token_id
    )
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels


if __name__ == "__main__":

    # fix the seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    args = get_args()

    # Load peft config for pre-trained checkpoint etc.

    if args.with_lora:
        # Load peft config for pre-trained checkpoint etc.
        peft_model_id = "results_large"
        config = PeftConfig.from_pretrained(peft_model_id)
        # load base LLM model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0}
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
    else:
        base_model_name_or_path = "results_base"
        # load base LLM model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name_or_path, device_map={"": 0}
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    # Load the Lora model
    model.eval()
    # Load dataset from the hub and get a sample
    dataset = load_dataset("samsum")
    sample = dataset["test"][randrange(len(dataset["test"]))]

    input_ids = tokenizer(
        sample["dialogue"], return_tensors="pt", truncation=True
    ).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9
    )
    print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

    print(
        f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}"
    )

    # Metric
    metric = evaluate.load("rouge")

    # load test dataset from distk
    test_dataset = load_from_disk("data/eval/").with_format("torch")

    # run predictions
    # this can take ~45 minutes
    predictions, references = [], []
    for sample in tqdm(test_dataset):
        p, l = evaluate_peft_model(sample)
        predictions.append(p)
        references.append(l)

    # compute metric
    rogue = metric.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    # print results
    print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
    print(f"rouge2: {rogue['rouge2']* 100:2f}%")
    print(f"rougeL: {rogue['rougeL']* 100:2f}%")
    print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

    # Rogue1: 50.386161%
    # rouge2: 24.842412%
    # rougeL: 41.370130%
    # rougeLsum: 41.394230%
