import os
import time
import logging
import random
import argparse
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset

# Import related functions and classes from other files
from model import load_model_and_tokenizer, Truncated_GaLoreAdamW
from dataloader import process_qa_samples, preprocess_data, collate_fn
from inference import save_best_model, validate

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with DoRA")
    # Model and tokenizer configuration
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Pre-trained model name or path")
    # Data paths
    parser.add_argument("--train_file", type=str, default="/path/to/Train.csv", help="Path to training data CSV")
    parser.add_argument("--val_file", type=str, default="/path/to/Val.csv", help="Path to validation data CSV")
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-7, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of the update matrix in LoRA")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.3, help="LoRA dropout probability")
    # Galore configuration
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    # Model save path
    parser.add_argument("--save_path", type=str, default="/path/to/save/best_model", help="Path to save the best model")
    # Other parameters
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    # Inference related parameters (can be reused)
    parser.add_argument("--input_files", type=str,
                        default="/path/to/val1.csv,/path/to/val2.csv",
                        help="Comma-separated list of evaluation file paths")
    parser.add_argument("--output_dir", type=str, default="/path/to/output/results",
                        help="Output directory to save evaluation results")
    parser.add_argument("--lr_description", type=str, default="3e-7", help="Learning rate description")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, tokenizer, train_loader, valid_loader, optimizer, criterion, num_epochs, save_path, device):
    best_val_loss = float("inf")
    total_batches = len(train_loader)
    print("Start Training!")
    logging.info("Start Training!")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        batch_times = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} Time: {batch_time:.4f} seconds, Loss: {loss.item():.4f}")
            logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} Time: {batch_time:.4f} seconds, Loss: {loss.item():.4f}")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_train_loss / total_batches
        avg_val_loss = validate(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epoch Time: {epoch_time:.2f} seconds, Average Batch Time: {avg_batch_time:.4f} seconds")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epoch Time: {epoch_time:.2f} seconds, Average Batch Time: {avg_batch_time:.4f} seconds, LR: {args.lr_description}")
        best_val_loss = save_best_model(model, tokenizer, epoch+1, best_val_loss, avg_val_loss, save_path)
    return model

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set environment variables (e.g., TOKENIZERS_PARALLELISM, etc.)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Login to Hugging Face (token can be configured)
    from huggingface_hub import login
    login(token="Your_Hugging_Face_Token")
    
    # Load tokenizer and model
    tokenizer, base_model = load_model_and_tokenizer(args.model_name)
    base_model.to(device)
    
    # Configure parameter groups (example: optimize some parameters with GaLore)
    galore_params = []
    target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for module_name, module in base_model.named_modules():
        if not hasattr(module, "weight"):
            continue
        if not any(key in module_name for key in target_modules_list):
            continue
        print(f"Enable GaLore for module: {module_name}")
        galore_params.append(module.weight)
    id_galore_params = [id(p) for p in galore_params]
    regular_params = [p for p in base_model.parameters() if id(p) not in id_galore_params]
    param_groups = [
        {'params': regular_params}, 
        {'params': galore_params, 
         'rank': args.rank, 
         'update_proj_gap': args.update_proj_gap, 
         'scale': args.galore_scale, 
         'proj_type': args.proj_type}
    ]
    
    optimizer = Truncated_GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Read data and construct DataLoader
    train_qa_samples, valid_qa_samples = process_qa_samples(args.train_file, args.val_file)
    train_dataset = Dataset.from_list(train_qa_samples).map(lambda ex: preprocess_data(ex, tokenizer), remove_columns=["question", "answer"])
    valid_dataset = Dataset.from_list(valid_qa_samples).map(lambda ex: preprocess_data(ex, tokenizer), remove_columns=["question", "answer"])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Start training
    trained_model = train(base_model, tokenizer, train_loader, valid_loader, optimizer, criterion, args.num_epochs, args.save_path, device)
