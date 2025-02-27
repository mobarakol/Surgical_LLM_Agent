import pandas as pd
import torch
import random
import numpy as np

def extract_question(text):
    return text.strip() if pd.notna(text) else ""

def extract_answer(text):
    return text.strip() if pd.notna(text) else ""

def generate1_SM(que):
    sm = (
        "Select a model to answer the following question:\n"
        f"'{que}'\n"
        "Select a model only from the following model list: Segment-Video, Segment-MRI, Detect-Instrument, Overlaying, Visual-Question-Answering to answer the question.\n"
        "Generate a shortest prompt for the chosen model. Your response should be like Model: ? Prompt: ? and no other words."
    )
    return sm

def generate2_SM(que):
    sm = (
        "Select models to answer the following question:\n"
        f"'{que}'\n"
        "Select appropriate models based on the question and select them only from the following model list: [Segment-Video, Segment-MRI, Detect-Instrument, Overlaying, Visual-Question-Answering]\n"
        "Generate shortest prompts for selected models. The format of your response should be like: 'Step1: Model: ? Prompt: ?...' Use as many steps as necessary, and no other words."
    )
    return sm

def process_qa_samples(train_file, val_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    for df, name in [(train_df, 'Train.csv'), (val_df, 'Val.csv')]:
        if 'Input' not in df.columns or 'Label' not in df.columns:
            print(f"CSV file {name} is missing 'Input' or 'Label' columns")
            return

    train_qa_samples = []
    for _, row in train_df.iterrows():
        question = extract_question(str(row['Input']))
        answer = extract_answer(str(row['Label']))
        if question and answer:
            if "Step1" in answer:
                question = generate2_SM(question)
            else:
                question = generate1_SM(question)
            train_qa_samples.append({"question": question, "answer": answer})

    valid_qa_samples = []
    for _, row in val_df.iterrows():
        question = extract_question(str(row['Input']))
        answer = extract_answer(str(row['Label']))
        if question and answer:
            if "Step1" in answer:
                question = generate2_SM(question)
            else:
                question = generate1_SM(question)
            valid_qa_samples.append({"question": question, "answer": answer})
    
    print("Train sample num:", len(train_qa_samples))
    print("Val sample num:", len(valid_qa_samples))
    if train_qa_samples:
        print("Example Train Sample:", train_qa_samples[0])
    if valid_qa_samples:
        print("Example Val Sample:", valid_qa_samples[0])
    
    return train_qa_samples, valid_qa_samples

def preprocess_data(example, tokenizer, max_length=196):
    input_text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)
    labels = inputs["input_ids"].copy()
    question_length = len(tokenizer(f"Question: {example['question']}\nAnswer:")["input_ids"]) - 1
    for i in range(len(labels)):
        if i < question_length or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    inputs["labels"] = labels
    return inputs

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
