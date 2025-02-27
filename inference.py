import os
import torch
import re
import time
import logging
import random
import numpy as np
import evaluate
from nltk.tokenize import word_tokenize

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

def format_data(sample):
    if "Step1" in sample[1]:
        system_message = generate2_SM(sample[0])
    else:
        system_message = generate1_SM(sample[0])
    return [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": sample[1]}
    ]

def custom_collate_fn(sample):
    return sample

def generate_answer(question, model, tokenizer):
    model.eval()
    input_text = f"Question: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    return answer

def extract_model(text, model_names):
    matches = re.findall(r'\b(?:' + '|'.join(map(re.escape, model_names)) + r')\b', text)
    return "|".join(matches) if matches else ""

def group_by_sentence_position(all_prompts, num_sentences):
    grouped_sentences = [[] for _ in range(num_sentences)]
    for prompts in all_prompts:
        sentences = prompts.split("|")
        for i in range(num_sentences):
            if i < len(sentences):
                grouped_sentences[i].append(sentences[i])
            else:
                grouped_sentences[i].append("")
    return grouped_sentences

def compute_metrics(grouped_pred_prompts, grouped_ans_prompts):
    rouge = evaluate.load("rouge")
    bleu_scores = []
    meteor_scores = []
    rouge_results = []
    for i in range(len(grouped_pred_prompts)):
        pred_group = grouped_pred_prompts[i]
        ans_group = grouped_ans_prompts[i]
        rouge_result = rouge.compute(predictions=pred_group, references=ans_group)
        bleu_score = evaluate.load("bleu").compute(predictions=pred_group, references=[[ans] for ans in ans_group])["bleu"]
        m_score = 0
        for ref, hypo in zip(ans_group, pred_group):
            ref_tokens = word_tokenize(ref)
            hypo_tokens = word_tokenize(hypo)
            m_score += 1.0  # 这里调用 meteor_score 或自定义实现
        meteor_avg = m_score / len(ans_group)
        bleu_scores.append(bleu_score)
        meteor_scores.append(meteor_avg)
        rouge_results.append(rouge_result)
    return rouge_results, bleu_scores, meteor_scores

def match_rate_per_Cat(pred_models_format, true_models_format):
    first_model_match_count = second_model_match_count = third_model_match_count = 0
    total_count = len(true_models_format)
    for pred, true in zip(pred_models_format, true_models_format):
        pred_models = pred.split("|")
        true_models = true.split("|")
        while len(pred_models) < len(true_models):
            pred_models.append(" ")
        if len(true_models) > 0 and pred_models[0] == true_models[0]:
            first_model_match_count += 1
        if len(true_models) > 1 and pred_models[1] == true_models[1]:
            second_model_match_count += 1
        if len(true_models) > 2 and pred_models[2] == true_models[2]:
            third_model_match_count += 1
    first_rate = (first_model_match_count / total_count * 100) if total_count > 0 else 0
    second_rate = (second_model_match_count / total_count * 100) if total_count > 0 else 0
    third_rate = (third_model_match_count / total_count * 100) if total_count > 0 else 0
    return first_rate, second_rate, third_rate

def f1_score_set(pred_list, true_list):
    pred_set = set(pred_list)
    true_set = set(true_list)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def evaluate_f1_by_selection_count(pred_models_format, true_models_format):
    two_model_scores = []
    three_model_scores = []
    for pred, true in zip(pred_models_format, true_models_format):
        pred_models = [m.strip() for m in pred.split("|")]
        true_models = [m.strip() for m in true.split("|")]
        while len(pred_models) < len(true_models):
            pred_models.append("")
        if len(true_models) == 2:
            score = f1_score_set(pred_models[:2], true_models[:2])
            two_model_scores.append(score)
        elif len(true_models) == 3:
            score = f1_score_set(pred_models[:3], true_models[:3])
            three_model_scores.append(score)
    avg_two_model_f1 = sum(two_model_scores) / len(two_model_scores) if two_model_scores else 0
    avg_three_model_f1 = sum(three_model_scores) / len(three_model_scores) if three_model_scores else 0
    return avg_two_model_f1, avg_three_model_f1

def save_best_model(model, tokenizer, epoch, best_loss, current_loss, save_path):
    new_save_path = f"{save_path}_{epoch}"
    os.makedirs(new_save_path, exist_ok=True)
    model.save_pretrained(new_save_path)
    tokenizer.save_pretrained(new_save_path)
    print(f"Current model saved at epoch {epoch} with val loss: {current_loss:.4f} in: {new_save_path}")
    if current_loss < best_loss:
        best_loss = current_loss
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Best model saved at epoch {epoch} with val loss: {best_loss:.4f}")
    return best_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)
