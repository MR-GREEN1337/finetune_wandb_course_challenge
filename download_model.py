#!/usr/bin/env python
"""
Direct download of TinyLlama model from HuggingFace
"""

import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Model to download
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = Path("checkpoints") / MODEL_NAME.replace("/", "-")

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {MODEL_NAME} directly from HuggingFace...")
    try:
        # Download model files
        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False
        )
        
        print(f"Model downloaded to {OUTPUT_DIR}")
        
        # Create lit_config.json if it doesn't exist
        config_file = OUTPUT_DIR / "lit_config.json"
        if not config_file.exists():
            # Try to extract information from config.json
            hf_config_file = OUTPUT_DIR / "config.json"
            if hf_config_file.exists():
                with open(hf_config_file, "r") as f:
                    hf_config: dict = json.load(f)
                
                # Create lit_config.json
                lit_config = {
                    "block_size": hf_config.get("max_position_embeddings", 2048),
                    "vocab_size": hf_config.get("vocab_size", 32000),
                    "n_layer": hf_config.get("num_hidden_layers", 22),
                    "n_head": hf_config.get("num_attention_heads", 32),
                    "dim": hf_config.get("hidden_size", 2048),
                    "rotary_percentage": 1.0,
                    "parallel_residual": False,
                    "bias": False,
                    "norm_eps": hf_config.get("rms_norm_eps", 1e-05)
                }
                
                with open(config_file, "w") as f:
                    json.dump(lit_config, f, indent=2)
                print(f"Created lit_config.json")
            else:
                print("Warning: Could not find config.json to extract model information")
                # Create a default lit_config.json
                default_config = {
                    "block_size": 2048,
                    "vocab_size": 32000,
                    "n_layer": 22,
                    "n_head": 32,
                    "dim": 2048
                }
                with open(config_file, "w") as f:
                    json.dump(default_config, f, indent=2)
                print("Created default lit_config.json")
        
        # Create lit_model.pth if it doesn't exist (copy from pytorch_model.bin)
        model_file = OUTPUT_DIR / "lit_model.pth"
        if not model_file.exists():
            hf_model_file = OUTPUT_DIR / "pytorch_model.bin"
            if hf_model_file.exists():
                shutil.copy2(hf_model_file, model_file)
                print(f"Created lit_model.pth from pytorch_model.bin")
            else:
                print("Warning: Could not find pytorch_model.bin to create lit_model.pth")
                
                # Check for sharded model files
                sharded_files = list(OUTPUT_DIR.glob("pytorch_model-*.bin"))
                if sharded_files:
                    print("Found sharded model files. These need to be merged.")
                    print("Please run the following manual steps:")
                    print("1. Install lit-gpt: pip install -e ./lit-gpt")
                    print("2. Run: python -m lit_gpt.scripts.convert_hf_checkpoint --checkpoint_dir checkpoints/TinyLlama-TinyLlama-1.1B-Chat-v1.0")
        
        print("Done!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()