#!/usr/bin/env python
"""
Pure PyTorch implementation of LoRA fine-tuning for math tutoring
Optimized for MacBook Pro M4 with MPS
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Custom LoRA implementation
class LoRALinear(nn.Module):
    """
    LoRA implemented in a dense layer
    """
    def __init__(
        self, 
        linear_layer: nn.Linear, 
        r: int = 8, 
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Store the original layer
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # LoRA parameters
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices - NOTE: These will be properly moved to the correct device later
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make sure LoRA weights are on the same device as input
        if self.lora_A.device != x.device:
            self.lora_A = nn.Parameter(self.lora_A.to(x.device))
            self.lora_B = nn.Parameter(self.lora_B.to(x.device))
            
        # Original linear layer path
        result = self.linear(x)
        
        # LoRA path
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        
        # Merge outputs
        return result + lora_output * self.scaling

def apply_lora_to_linear_layers(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: List[str] = None,
    device = "cpu"
) -> nn.Module:
    """
    Apply LoRA to specific layers in the model
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
    
    # Track replaced modules
    replaced_modules = []
    
    # Replace linear layers with LoRA layers
    for name, module in model.named_modules():
        # Check if the module is a target and is a linear layer
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            # Get the parent module and attribute name
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else get_module(model, parent_name)
            
            # Create a LoRA layer
            lora_layer = LoRALinear(
                linear_layer=module,
                r=r,
                alpha=alpha,
                dropout=dropout
            )
            
            # Move the LoRA parameters to the same device as the model
            lora_layer.to(device)
            
            # Replace the original layer with LoRA layer
            setattr(parent, attr_name, lora_layer)
            replaced_modules.append(name)
    
    print(f"Applied LoRA to {len(replaced_modules)} modules: {replaced_modules}")
    return model

def get_module(model, path):
    """Get a module from the model given its path"""
    parts = path.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    Freeze all parameters except LoRA parameters
    """
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

# Configuration
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    local_model_path: str = "checkpoints/TinyLlama-TinyLlama-1.1B-Chat-v1.0"
    out_dir: str = "out/math-tutor"
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Will be set based on the flags below
    
    # Target module flags
    lora_query: bool = True
    lora_key: bool = True
    lora_value: bool = True
    lora_projection: bool = True
    lora_mlp: bool = True
    lora_head: bool = True
    
    # Training
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # Using ratio instead of steps
    max_steps: Optional[int] = None
    
    # Mac optimization
    use_fp32: bool = True  # Mac MPS works better with FP32 for small models
    
    # Data
    data_dir: str = "data/math_tutoring"
    max_seq_length: int = 1024
    
    # System
    seed: int = 42
    device: str = None  # Will be set automatically
    
    # W&B
    wandb_project: str = "math-tutor-llm"
    wandb_run_name: Optional[str] = None
    wandb_log_model: str = "all"  # Options: "all", "end", None

class MathTutoringDataset(Dataset):
    def __init__(self, data_file, max_seq_length):
        super().__init__()
        self.data = torch.load(data_file)
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels = torch.tensor(item["labels"], dtype=torch.long)
        
        # Concatenate input and labels
        combined = torch.cat([input_ids, labels])
        
        # If too long, truncate from the beginning
        if len(combined) > self.max_seq_length:
            combined = combined[-self.max_seq_length:]
        
        # Create attention mask (1 for all tokens)
        attention_mask = torch.ones_like(combined)
        
        # Create labels (-100 for input_ids, actual labels for output)
        input_len = min(len(input_ids), self.max_seq_length)
        new_labels = torch.full_like(combined, -100)
        
        if len(combined) > input_len:
            new_labels[input_len:] = combined[input_len:]
        
        return {
            "input_ids": combined,
            "attention_mask": attention_mask,
            "labels": new_labels
        }

# Custom collate function to handle variable-sized tensors
def custom_collate_fn(batch):
    """
    Custom collate function that handles variable sized tensors by padding
    """
    # Find max length in the batch
    max_length = max(item["input_ids"].size(0) for item in batch)
    
    # Initialize tensors
    input_ids = []
    attention_masks = []
    labels = []
    
    # Pad sequences
    for item in batch:
        # Get tensors
        input_id = item["input_ids"]
        attention_mask = item["attention_mask"]
        label = item["labels"]
        
        # Calculate padding
        pad_length = max_length - input_id.size(0)
        
        if pad_length > 0:
            # Pad input_ids with pad token (usually 0)
            input_id = torch.cat([input_id, torch.zeros(pad_length, dtype=torch.long)])
            
            # Pad attention_mask with 0s (don't attend to padding)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
            
            # Pad labels with -100 (ignore in loss calculation)
            label = torch.cat([label, torch.full((pad_length,), -100, dtype=torch.long)])
        
        # Add to lists
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)
    
    # Stack tensors
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels)
    }

def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,  # Use 0 for Mac to avoid multiprocessing issues
        collate_fn=custom_collate_fn
    )

def setup_wandb(config):
    """Initialize W&B run"""
    run_name = config.wandb_run_name or f"math-tutor-{time.strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=vars(config)
    )
    
    # Try to load model config for logging
    try:
        config_path = Path(config.local_model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
            
            # Log key model info
            wandb.config.update({
                "model_name": config.model_name,
                "vocab_size": model_config.get("vocab_size", "unknown"),
                "hidden_size": model_config.get("hidden_size", "unknown"),
                "num_layers": model_config.get("num_hidden_layers", "unknown"),
                "num_heads": model_config.get("num_attention_heads", "unknown"),
            })
    except Exception as e:
        print(f"Warning: Could not load model config for W&B logging: {e}")
    
    return wandb.run

def move_batch_to_device(batch, device):
    """Move a batch of data to the specified device"""
    return {k: v.to(device) for k, v in batch.items()}

def train():
    """Run the training process using pure PyTorch"""
    # Initialize config
    config = TrainingConfig()
    
    # Set device for Mac
    if torch.backends.mps.is_available():
        config.device = "mps"
        print("Using MPS (Metal Performance Shaders) for training")
    elif torch.cuda.is_available():
        config.device = "cuda"
        print("Using CUDA for training")
    else:
        config.device = "cpu"
        print("Using CPU for training (this will be slow)")
    
    device = torch.device(config.device)
    
    # Set random seed
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    
    # Create target modules list based on LoRA flags
    config.target_modules = []
    if config.lora_query:
        config.target_modules.append("q_proj")
    if config.lora_key:
        config.target_modules.append("k_proj")
    if config.lora_value:
        config.target_modules.append("v_proj")
    if config.lora_projection:
        config.target_modules.append("o_proj")
    if config.lora_mlp:
        config.target_modules.extend(["gate_proj", "up_proj", "down_proj"])
    if config.lora_head:
        config.target_modules.append("lm_head")
    
    # Setup wandb
    wandb_run = setup_wandb(config)
    
    # Setup output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(out_dir / "train_config.json", "w") as f:
        # Convert list to string for JSON serialization
        config_dict = {k: v if not isinstance(v, List) else str(v) for k, v in vars(config).items()}
        json.dump(config_dict, f, indent=2)
    
    # Load model and tokenizer
    print(f"Loading model from {config.local_model_path}")
    
    # For Mac, we'll load the model in FP32 by default
    dtype = torch.float32 if config.use_fp32 else torch.float16
    
    # First load the model entirely to CPU to avoid MPS memory issues
    model = AutoModelForCausalLM.from_pretrained(
        config.local_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    # Apply LoRA to the model while it's on CPU
    model = apply_lora_to_linear_layers(
        model,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules,
        device="cpu"  # Apply on CPU first
    )
    
    # Mark only LoRA as trainable
    mark_only_lora_as_trainable(model)
    
    # Now move model to the device
    print(f"Moving model to {config.device}")
    model = model.to(device)
    
    # Setup optimizer - only training LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    
    print(f"Number of trainable parameters: {trainable_param_count} ({trainable_param_count/total_param_count:.2%} of total)")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Try to load datasets
    try:
        print(f"Loading datasets from {config.data_dir}")
        train_dataset = MathTutoringDataset(
            f"{config.data_dir}/train.pt",
            config.max_seq_length
        )
        
        val_dataset = MathTutoringDataset(
            f"{config.data_dir}/test.pt",
            config.max_seq_length
        )
        
        print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
        
        # Create data loaders with custom collate function
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=config.micro_batch_size,
            shuffle=True
        )
        
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=config.micro_batch_size,
            shuffle=False
        )
        
        # Debug: Test data loading (move batch to device as we'll do in training)
        print("Testing data loading...")
        for batch_idx, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)
            print(f"Batch {batch_idx} loaded successfully and moved to {device}")
            print(f"- input_ids shape: {batch['input_ids'].shape}")
            print(f"- attention_mask shape: {batch['attention_mask'].shape}")
            print(f"- labels shape: {batch['labels'].shape}")
            if batch_idx == 0:
                break
                
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    # Compute steps
    if config.max_steps is None:
        config.max_steps = len(train_dataloader) * config.epochs
    
    # Setup LR scheduler with warmup ratio
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.learning_rate,
        total_steps=config.max_steps,
        pct_start=config.warmup_ratio,
        div_factor=25.0,
        final_div_factor=10000.0,
        anneal_strategy="cos"
    )
    
    # Training loop
    model.train()
    step_count = 0
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        # Training
        epoch_loss = 0
        epoch_step = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            if step_count >= config.max_steps:
                break
                
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                
                # Optimizer and scheduler step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                
                # Log metrics
                wandb.log({
                    "train/loss": loss.item() * config.gradient_accumulation_steps,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": step_count,
                    "train/epoch": epoch + (batch_idx / len(train_dataloader))
                })
                
                if step_count % 5 == 0:
                    print(f"Step {step_count}/{config.max_steps} - Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            epoch_step += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = move_batch_to_device(batch, device)
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1
        
        val_loss /= max(1, val_steps)  # Avoid division by zero
        
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": epoch_loss / max(1, epoch_step),
            "val/loss": val_loss,
            "val/epoch": epoch + 1
        })
        
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss / max(1, epoch_step):.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = out_dir / f"checkpoint-epoch-{epoch+1}.bin"
        
        # Save state dict (CPU for compatibility)
        torch.save({
            "model_state_dict": {k: v.to('cpu') for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch,
            "config": vars(config)
        }, checkpoint_path)
        
        if config.wandb_log_model in ["all", "checkpoint"]:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch+1}",
                type="model",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
        
        model.train()
    
    # Save final model
    final_model_path = out_dir / "final_model.bin"
    
    # Save state dict (to CPU for compatibility)
    torch.save({
        "model_state_dict": {k: v.to('cpu') for k, v in model.state_dict().items()},
        "config": vars(config)
    }, final_model_path)
    
    # Save in HF format for easy loading
    try:
        # Move to CPU for saving
        model_to_save = model.to('cpu')
        model_to_save.save_pretrained(out_dir / "hf_model")
    except Exception as e:
        print(f"Warning: Could not save model in HF format: {e}")
    
    if config.wandb_log_model in ["all", "end"]:
        artifact = wandb.Artifact(
            name="model-final",
            type="model",
            description="Final trained model"
        )
        artifact.add_file(str(final_model_path))
        wandb.log_artifact(artifact)
    
    # Close wandb
    wandb.finish()
    
    print(f"Training complete. Model saved to {out_dir}")

if __name__ == "__main__":
    train()