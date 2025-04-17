#!/usr/bin/env python
"""
Math Tutor LLM Server with W&B Logging - Loading directly from final_model.bin
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import uuid

import torch
import uvicorn
import gradio as gr
import wandb
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# API Models
class TutorRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    session_id: Optional[str] = None
    feedback: Optional[str] = None
    log_to_wandb: bool = True
    math_topic: Optional[str] = None

class TutorResponse(BaseModel):
    response: str
    session_id: str
    request_time: float

# Global variables
app = FastAPI(title="Math Tutor LLM")
wandb_run = None
model = None
tokenizer = None
device = None
generation_config = None

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
CHECKPOINT_PATH = Path("out/math-tutor/final_model.bin")  # PyTorch state dict
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_BASE_MODEL = Path("checkpoints/TinyLlama-TinyLlama-1.1B-Chat-v1.0")

WANDB_PROJECT = "math-tutor-llm-inference"
INSTRUCTION_TEMPLATE = """You are a helpful and knowledgeable math tutor. 
Your goal is to help students understand math concepts clearly and develop problem-solving skills.
Provide step-by-step explanations, use analogies when helpful, and encourage critical thinking.
"""

# Custom LoRA implementation (simplified version for inference only)
class LoRALinear(torch.nn.Module):
    def __init__(
        self, 
        linear_layer, 
        r=8, 
        alpha=16,
        dropout=0.0,
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
        
        # LoRA matrices
        self.lora_A = torch.nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = torch.nn.Parameter(torch.zeros(r, self.out_features))
        
        # Optional dropout
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
    
    def forward(self, x):
        # Make sure LoRA weights are on the same device as input
        if self.lora_A.device != x.device:
            self.lora_A = torch.nn.Parameter(self.lora_A.to(x.device))
            self.lora_B = torch.nn.Parameter(self.lora_B.to(x.device))
            
        # Original linear layer path
        result = self.linear(x)
        
        # LoRA path
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        
        # Merge outputs
        return result + lora_output * self.scaling

def apply_lora_to_linear_layers(
    model: torch.nn.Module,
    checkpoint_state_dict: Dict,
    target_modules: List[str] = None,
    device = "cpu"
) -> torch.nn.Module:
    """
    Apply LoRA layers from a checkpoint to a model
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
    
    # Track replaced modules
    replaced_modules = []
    
    # Find LoRA modules in checkpoint
    lora_state_dict = {}
    for key in checkpoint_state_dict.keys():
        if ".lora_A" in key or ".lora_B" in key:
            lora_state_dict[key] = checkpoint_state_dict[key]
    
    # If no LoRA weights found, return model as is
    if not lora_state_dict:
        print("No LoRA weights found in checkpoint")
        return model
    
    # Replace linear layers with LoRA layers
    for name, module in model.named_modules():
        # Check if the module is a target and is a linear layer
        if any(target in name for target in target_modules) and isinstance(module, torch.nn.Linear):
            # Get the parent module and attribute name
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else get_module(model, parent_name)
            
            # Extract LoRA parameters for this module
            module_prefix = f"{name}."
            has_lora_weights = False
            
            for key in lora_state_dict:
                if module_prefix in key:
                    has_lora_weights = True
                    break
            
            if not has_lora_weights:
                continue
            
            # Create LoRA layer
            lora_layer = LoRALinear(
                linear_layer=module,
                r=8,
                alpha=16,
                dropout=0.0
            )
            
            # Replace the original layer with LoRA layer
            setattr(parent, attr_name, lora_layer)
            replaced_modules.append(name)
    
    # Now load LoRA weights
    missing, unexpected = model.load_state_dict(checkpoint_state_dict, strict=False)
    
    print(f"Applied LoRA to {len(replaced_modules)} modules")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    return model

def get_module(model, path):
    """Get a module from the model given its path"""
    parts = path.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current

def format_prompt(question: str, topic: Optional[str] = None) -> str:
    """Format a question for the math tutor model"""
    # Format the prompt with chat template
    if topic and topic.lower() != "general":
        prompt = f"{INSTRUCTION_TEMPLATE}\nTopic: {topic}\nQuestion: {question}"
    else:
        prompt = f"{INSTRUCTION_TEMPLATE}\nQuestion: {question}"
        
    # Format for chat
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def init_wandb():
    """Initialize W&B for logging inference"""
    global wandb_run
    if wandb_run is None:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"inference-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_name": "math-tutor-final-model",
                "base_model": str(BASE_MODEL),
            }
        )
    return wandb_run

def log_to_wandb(
    request: TutorRequest, 
    response: str, 
    request_time: float,
    session_id: str
):
    """Log inference to W&B"""
    if not request.log_to_wandb:
        return
    
    # Initialize W&B if not already done
    if wandb_run is None:
        init_wandb()
    
    # Log the inference
    wandb.log({
        "prompt": request.prompt,
        "response": response,
        "session_id": session_id,
        "request_time": request_time,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "max_new_tokens": request.max_new_tokens,
        "math_topic": request.math_topic or "general",
        "feedback": request.feedback or "none"
    })
    
    # If feedback is provided, log it to a custom table
    if request.feedback:
        feedback_table = wandb.Table(
            columns=["session_id", "prompt", "response", "feedback"]
        )
        feedback_table.add_data(
            session_id, 
            request.prompt, 
            response, 
            request.feedback
        )
        wandb.log({"feedback": feedback_table})

@app.on_event("startup")
async def startup_event():
    """Initialize the model and tokenizer on startup"""
    global model, tokenizer, device, generation_config
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Check if checkpoint exists
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    # Load base model
    print(f"Loading base model: {LOCAL_BASE_MODEL if LOCAL_BASE_MODEL.exists() else BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_BASE_MODEL if LOCAL_BASE_MODEL.exists() else BASE_MODEL,
        torch_dtype=torch.float32,  # Use fp32 for Mac
        device_map={"": "cpu"}  # First load to CPU
    )
    
    # Load checkpoint
    print(f"Loading fine-tuned weights from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Apply LoRA layers and load weights
    model = apply_lora_to_linear_layers(model, state_dict)
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_BASE_MODEL if LOCAL_BASE_MODEL.exists() else BASE_MODEL
        )
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
    
    # Default generation config
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    print("Model and tokenizer loaded successfully")
    
    # Initialize W&B
    init_wandb()

@app.post("/tutor")
async def tutor(
    request: TutorRequest, 
    background_tasks: BackgroundTasks
) -> TutorResponse:
    """Process a tutoring request"""
    start_time = time.time()
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Format the prompt
    formatted_prompt = format_prompt(request.prompt, request.math_topic)
    
    # Tokenize the prompt
    input_ids = tokenizer(
        formatted_prompt, 
        return_tensors="pt"
    ).input_ids.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the model's response (remove the prompt)
    try:
        # Try to extract the assistant's part of the response
        response = output_text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
    except:
        # Fallback: Just return everything after the prompt
        response = output_text[len(formatted_prompt):]
        # Try to clean up any remaining special tokens
        response = response.replace("<|im_end|>", "").strip()
    
    request_time = time.time() - start_time
    
    # Log to W&B in the background
    background_tasks.add_task(
        log_to_wandb,
        request=request,
        response=response,
        request_time=request_time,
        session_id=session_id
    )
    
    return TutorResponse(
        response=response,
        session_id=session_id,
        request_time=request_time
    )

@app.post("/feedback")
async def feedback(
    session_id: str,
    feedback_text: str,
    background_tasks: BackgroundTasks
):
    """Log feedback for a tutoring session"""
    background_tasks.add_task(
        lambda: wandb.log({
            "feedback_standalone": {
                "session_id": session_id,
                "feedback": feedback_text,
                "timestamp": time.time()
            }
        })
    )
    return {"status": "Feedback recorded"}

# Gradio Interface
def create_gradio_interface():
    """Create a Gradio interface for the math tutor"""
    with gr.Blocks(title="Math Tutor LLM") as interface:
        session_id = gr.State(value=str(uuid.uuid4()))
        
        gr.Markdown("# Math Tutor LLM")
        gr.Markdown("Ask any math question and get step-by-step help from your AI math tutor.")
        
        with gr.Row():
            with gr.Column(scale=4):
                topic = gr.Dropdown(
                    choices=[
                        "General",
                        "Algebra",
                        "Calculus",
                        "Geometry",
                        "Trigonometry",
                        "Statistics",
                        "Probability",
                        "Number Theory",
                        "Linear Algebra"
                    ],
                    label="Topic",
                    value="General"
                )
                
                question = gr.Textbox(
                    lines=4,
                    placeholder="Type your math question here...",
                    label="Your Question"
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=1024,
                        value=512,
                        step=50,
                        label="Max Response Length"
                    )
                
                submit_btn = gr.Button("Ask Tutor", variant="primary")
                clear_btn = gr.Button("Clear")
                
            with gr.Column(scale=6):
                response = gr.Markdown(label="Tutor's Response")
                
                with gr.Row():
                    feedback_text = gr.Textbox(
                        lines=2,
                        placeholder="Was this explanation helpful? Provide feedback...",
                        label="Feedback"
                    )
                    feedback_btn = gr.Button("Submit Feedback")
        
        def submit_question(topic_val, question_val, temp_val, max_tokens_val, session_id_val):
            if not question_val.strip():
                return "Please type a question.", session_id_val
            
            # Use API endpoint
            import requests
            
            # Format the request
            request_data = {
                "prompt": question_val,
                "temperature": temp_val,
                "max_new_tokens": max_tokens_val,
                "session_id": session_id_val,
                "math_topic": topic_val.lower() if topic_val != "General" else None
            }
            
            # Send request to local API
            response = requests.post(
                "http://localhost:8000/tutor",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["response"], data["session_id"]
            else:
                return f"Error: {response.text}", session_id_val
        
        def submit_user_feedback(feedback_val, session_id_val):
            if not feedback_val.strip():
                return
            
            # Send feedback to API
            import requests
            
            response = requests.post(
                "http://localhost:8000/feedback",
                json={
                    "session_id": session_id_val,
                    "feedback_text": feedback_val
                }
            )
            
            return
        
        def clear_interface():
            return "", "", gr.update(value="")
        
        # Wire up the interface
        submit_btn.click(
            submit_question,
            inputs=[topic, question, temperature, max_tokens, session_id],
            outputs=[response, session_id]
        )
        
        feedback_btn.click(
            submit_user_feedback,
            inputs=[feedback_text, session_id],
            outputs=[]
        )
        
        clear_btn.click(
            clear_interface,
            inputs=[],
            outputs=[question, response, feedback_text]
        )
        
    return interface

# Mount Gradio app to FastAPI
gr_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gr_app, path="/")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)