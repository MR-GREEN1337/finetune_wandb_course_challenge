# Math Tutor LLM - Weights & Biases Course Submission

This project demonstrates fine-tuning a smaller LLM (TinyLlama-1.1B) to act as a math tutor, with full Weights & Biases integration for experiment tracking. The model is fine-tuned to provide step-by-step explanations for math problems across various topics.

## Project Overview

- **Task**: Fine-tune an LLM to act as a math tutor
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: Custom dataset combining math problems from various sources
- **Tracking**: Full W&B integration for both training and inference
- **Deployment**: FastAPI + Gradio interface

## Files in this Repository

- `download_model.py` - Downloads the base TinyLlama model
- `prepare_dataset.py` - Creates a math tutoring dataset from multiple sources
- `finetune.py` - Runs LoRA fine-tuning with W&B logging
- `server.py` - Serves the model with FastAPI and Gradio UI
- `README.md` - This file

## How to Use

### Setup (Mac)

```bash
# Clone this repository
git clone https://github.com/your-username/math-tutor-llm
cd math-tutor-llm


# Login to Weights & Biases
wandb login
```

### Download the Model

```bash
python download_model.py
```

### Prepare the Dataset

```bash
python prepare_dataset.py
```

### Fine-tune the Model

```bash
python finetune.py
```

This will start the fine-tuning process and create a W&B run with:
- Model architecture details
- Training hyperparameters
- Training and validation loss tracking
- Model checkpoints

### Start the Server

```bash
python server.py
```

This will:
1. Start a FastAPI server on http://localhost:8000
2. Initialize a Gradio interface at http://localhost:8000
3. Start logging inferences to W&B

## Weights & Biases Integration

### Training Tracking

The training script (`finetune.py`) tracks:
- Model architecture details
- Training configuration
- Learning rate schedule
- Training and validation loss
- Model checkpoints (saved as artifacts)

### Inference Tracking

The server (`server.py`) tracks:
- User queries
- Model responses
- Response time
- User feedback
- Session information

### Project Report

The W&B report includes:
1. **Project Overview** - Description of the math tutoring task
2. **Model Selection** - Why TinyLlama was chosen for a Mac-compatible solution
3. **Dataset Creation** - How the math tutoring dataset was created
4. **Hyperparameter Optimization** - Experiments with different LoRA configurations
5. **Training Results** - Analysis of training and validation loss
6. **Inference Analysis** - Performance on different math topics
7. **User Feedback Analysis** - Analysis of user satisfaction based on feedback
8. **Future Improvements** - Ideas for enhancing the model

## Link to W&B Report

[wandb report](https://wandb.ai/islam-hachimi2003-student/math-tutor-llm/reports/Finetuning-TinyLlama-1-1B-Chat-v1-0--VmlldzoxMjM0NTM3Mg?accessToken=67civ1b5wg06qrx1mbz9mdsxjro08zyauju49f4d1y2xj4adra1hjgopfdw6asvx)