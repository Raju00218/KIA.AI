import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse  # Added this import
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast  # Phase 3: Tokenizer
from datasets import load_dataset  # Phase 3: Real Data
# 1. THE AI BRAIN: 2 Billion Parameter Transformer Architecture (from scratch)
# ==============================================================================

class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None, past_key_value=None):
        B, T, C = x.size()
        
        # Calculate Q, K, V and split into heads
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Phase 4: KV Caching: Append past keys and values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
            
        present_key_value = (k, v)
        
        T_q = q.size(-2)
        T_k = k.size(-2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask context
        if mask is None and T_q > 1:
            mask = torch.tril(torch.ones(T_q, T_k)).view(1, 1, T_q, T_k).to(x.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Aggregate values
        out = torch.matmul(attn_weights, v)
        
        # Re-assemble heads and project
        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.W_o(out), present_key_value

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # SwiGLU or GELU activation is common here for large models
        return self.w2(F.gelu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = Attention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None, past_key_value=None):
        # Pre-LN architecture with KV cache support
        attn_out, present_key_value = self.attn(self.norm1(x), mask=mask, past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present_key_value

class KiaAI(nn.Module):
    def __init__(self, vocab_size, d_model=2560, num_layers=32, num_heads=32, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        
        # Token and Positional Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Layers
        d_ff = 4 * d_model # Standard multiplier
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        
    def forward(self, idx, past_key_values=None):
        B, T = idx.size()
        device = idx.device
        
        # If caching, we only process the new token. Calculate the real offset sequence length.
        seq_len = T if past_key_values is None else T + past_key_values[0][0].shape[-2]
        
        if past_key_values is None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
        else:
            pos = torch.arange(seq_len - 1, seq_len, dtype=torch.long, device=device)
            
        # Add embeddings
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, past_key_value=past_kv)
            present_key_values.append(present_kv)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, present_key_values
        
    def generate(self, idx, max_new_tokens):
        # Phase 4: KV Caching Implemented: Transforms O(N^2) complexity to O(N) by saving past keys/values
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is None:
                # First pass: process full sequence
                idx_cond = idx[:, -2048:] 
            else:
                # Subsequent passes: only process the newest generated token
                idx_cond = idx[:, -1:] 
                
            logits, past_key_values = self(idx_cond, past_key_values=past_key_values)
            
            logits = logits[:, -1, :] # Focus on last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    print("Welcome to KIA AI - 2B Parameter Initialization")
    print("WARNING: Allocating a 2B parameter model requires 8GB+ of VRAM just for weights.")
    
def main():
    # Phase 3: We switch from a hardcoded vocab size to the exact vocab size of our Tokenizer
    # GPT-2 Tokenizer is standard for English text.
    print("Loading Tokenizer...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # GPT-2 does not have a padding token by default, so we use the EOS (End of Sentence) token
        tokenizer.pad_token = tokenizer.eos_token 
        vocab_size = len(tokenizer) # Use len() to account for added tokens if any
        print(f"Tokenizer loaded! Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer (are you connected to the internet?): {e}")
        return
    
    try:
        # Initialize the model on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing KIA AI on {device}...")
        
        # Phase 4: Implement Half-Precision to cut VRAM requirements in half (16-bit instead of 32-bit floats)
        # We do this BEFORE creating the model so the initial memory spike is instantly halved.
        if device == 'cuda':
            print("Enabling bfloat16 half-precision optimizations...")
            torch.set_default_dtype(torch.bfloat16)
        
        # Instantiate Model
        # Phase 4: Scaling up! Using 1.6-Billion Parameter structure (Max safe size for 24GB L4 GPU)
        model = KiaAI(vocab_size=vocab_size, d_model=2048, num_layers=32, num_heads=32)

        model.to(device)
        print("Model initialized successfully (2B Parameter Scaled Architecture)!")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="KIA AI Backend")
    parser.add_argument('--train', action='store_true', help="Run the model training loop")
    parser.add_argument('--infer', action='store_true', help="Load saved weights and generate text")
    parser.add_argument('--serve', action='store_true', help="Run the FastAPI inference server to connect to React")
    parser.add_argument('--prompt', type=str, default="Once upon a time, there was a little", help="Custom text for the AI to complete")
    parser.add_argument('--save_dir', type=str, default='.', help="Folder path to save/load the model weights")
    parser.add_argument('--ngrok_token', type=str, default=None, help="Ngrok Auth Token for exposing the API")
    
    # In Jupyter/Colab environments, sys.argv includes an undocumented '-f' flag that crashes standard parsing.
    # To fix this, we parse with parse_known_args() and explicitly check if '--train' was provided in sys.argv.
    import sys
    args, unknown = parser.parse_known_args()

    # Note: Using '--train' in sys.argv check bypasses notebook default kernels overriding args.train
    if '--train' in sys.argv or args.train or ('--train' not in sys.argv and not args.train and not args.infer and not args.serve): # Default to train
        print("\n--- Starting KIA AI Model Training ---")
        train_model(model, device, args.save_dir, tokenizer)
    elif '--infer' in sys.argv or args.infer:
        print("\n--- Loading KIA AI Model for Inference ---")
        infer_model(model, device, args.save_dir, tokenizer, args.prompt)
    elif '--serve' in sys.argv or args.serve:
        print("\n--- Starting KIA AI Inference Server ---")
        serve_model(model, device, args.save_dir, tokenizer, args.ngrok_token)

def train_model(model, device, save_dir, tokenizer):
    """
    Core PyTorch Training Loop for the KIA AI model.
    In Phase 3, this is where we feed a dataset (like TinyStories) to adjust weights.
    In Phase 3, this is where we feed a dataset (like TinyStories) to adjust weights.
    """
    import os
    import json
    import re
    
    def clean_text(text):
        """
        Stage 1 Data Cleaning Pipeline:
        Removes non-ASCII chars and excessive whitespace to prevent model hallucinations.
        """
        # Remove anything that isn't a basic English letter, number, or common punctuation
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    
    # --- PHASE 6/9: ADVANCED TRAINING STABILITY ---
    max_lr = 3e-4 # Very high for core learning, but needs warmup!
    min_lr = 3e-5 # Cooldown speed
    warmup_steps = 2000
    total_steps = 200000
    
    # 1. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=(device=='cuda'), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    # 2. Setup 3-Stage Learning Rate Scheduler (Warmup -> Core -> Cooldown)
    def get_lr(step):
        # 1. Warmup
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        # 3. Cooldown (Wait for 150k steps before aggressively cooling down)
        if step > 150000:
            decay_ratio = (step - 150000) / (total_steps - 150000)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))
        # 2. Core Training
        return max_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    
    # Phase 6: Warm Start Initialization
    save_path = os.path.join(save_dir, 'kia_ai_2b.pth')
    state_path = os.path.join(save_dir, 'training_state.json')
    start_step = 0
    
    if os.path.exists(save_path):
        print(f"--- WARM START: Loading existing knowledge from {save_path} ---")
        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            print("Successfully recovered previous neural connections!")
            
            # Phase 8: Load the last saved step if available
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    start_step = state.get('step', 0)
                    print(f"Resuming from Step {start_step}...")
        except Exception as e:
            print(f"Warning: Could not warm start. Starting from scratch. Error: {e}")
            
    # 2. Phase 6 Dataset Loader (Streaming from HuggingFace) - CLEAN DATA ONLY
    
    print("Loading TinyStories dataset (100% of data for Stage 1 Pre-training)...")
    stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    combined_dataset = stories
        
    data_iterator = iter(combined_dataset)
    
    def get_batch():
        # Grab a batch of 1 story (to save VRAM on the L4 GPU)
        texts = []
        for _ in range(1):
            try:
                story = next(data_iterator)["text"]
                # Apply our rigorous text cleaning pipeline before the model sees it!
                cleaned_story = clean_text(story)
                texts.append(cleaned_story)
            except StopIteration:
                pass # Handled for simplicity during testing
        
        # Translate the English strings into Token IDs (Math)
        # We truncate to 64 context length (Down from 128 to save VRAM)
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=65)
        input_ids = tokens["input_ids"].to(device)
        
        # X: Input tokens [0 to 127]
        # Y: Target tokens [1 to 128] (shifted by 1 to predict the NEXT word)
        X = input_ids[:, :-1] 
        Y = input_ids[:, 1:]
        return X, Y

    iterations = total_steps # Phase 9: Using user-defined total steps (200k)
    accumulation_steps = 4 
    print(f"Beginning 3-Stage training loop for {iterations} iterations on {device}...")
    print(f"Schedule: 0-2k Warmup | 2k-150k Core Learning | 150k-200k Cooldown")
    print(f"Using Gradient Accumulation: Batch Size 1 x {accumulation_steps} Steps = Effective Batch Size 4")
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # Track loss to prevent "gibberish" overfitting
    best_loss = float('inf')
    loss_history = []
    
    try:
        for step in range(start_step, iterations):
            # Sample a micro-batch of data
            xb, yb = get_batch()
            
            # Forward pass: compute predictions
            logits, _ = model(xb)
            
            # Loss computation requires reshaping logits to (B*T, C) and targets to (B*T)
            B, T, C = logits.shape
            logits_reshaped = logits.reshape(B*T, C)
            targets = yb.reshape(B*T)
            
            # Scale the loss since we are accumulating gradients
            loss = loss_fn(logits_reshaped, targets) / accumulation_steps
            
            # Backward pass: compute gradients (accumulates them)
            loss.backward()
            
            # Step: update weights only every 'accumulation_steps'
            if (step + 1) % accumulation_steps == 0:
                # Gradient Clipping: The ultimate defense against model explosion!
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step() # Apply the 3-Stage LR schedule
                optimizer.zero_grad(set_to_none=True)
                
                # Aggressively clear VRAM cache to prevent OOM spikes during optimizer step
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Record loss history for gibberish prevention
            current_loss = loss.item() * accumulation_steps
            loss_history.append(current_loss)
            if len(loss_history) > 1000:
                loss_history.pop(0)
                
            if step % 100 == 0 or step == iterations - 1:
                current_lr = scheduler.get_last_lr()[0]
                avg_recent_loss = sum(loss_history[-100:]) / min(len(loss_history), 100)
                print(f"Step {step:6d} | LR: {current_lr:.2e} | Micro-Loss: {current_loss:.4f} | Avg-100-Loss: {avg_recent_loss:.4f}")

            # --- Phase 9: Auto-Save Checkpoint + Overfitting Monitor ---
            if (step + 1) % 5000 == 0:
                avg_recent_loss = sum(loss_history[-100:]) / min(len(loss_history), 100)
                
                # Check for model collapse/explosion
                if avg_recent_loss > best_loss * 2 and best_loss < 5.0:
                     print(f"\n🚨 [DANGER: OVERFITTING DETECTED] The loss has spiked severely compared to the best loss ({best_loss:.4f}).")
                     print("The model is starting to memorize noise (resulting in gibberish).")
                     print("Triggering EARLY STOPPING to save the brain.")
                     
                     # Revert to the best known state
                     backup_path = os.path.join(save_dir, 'kia_ai_2b_best.pth')
                     if os.path.exists(backup_path):
                         print("Reverting to the last stable checkpoint...")
                         model.load_state_dict(torch.load(backup_path))
                     break # Exit the training loop
                     
                # Save the new best known state
                if avg_recent_loss < best_loss:
                    best_loss = avg_recent_loss
                    backup_path = os.path.join(save_dir, 'kia_ai_2b_best.pth')
                    torch.save(model.state_dict(), backup_path)
                    
                print(f"\n[CHECKPOINT] Saving current training step {step+1}...")
                torch.save(model.state_dict(), save_path)
                with open(state_path, 'w') as f:
                    json.dump({'step': step + 1}, f)
    
    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPT] Training stopped at step {step}. Saving weights...")
        torch.save(model.state_dict(), save_path)
        with open(state_path, 'w') as f:
            json.dump({'step': step}, f)
        print(f"Model and state saved to {save_dir}. You can resume exactly from here!")
        return # Exit gracefully
    except Exception as e:
        print(f"\n\n[CRASH] Training failed: {e}. Attempting emergency save...")
        torch.save(model.state_dict(), save_path)
        with open(state_path, 'w') as f:
            json.dump({'step': step}, f)
        raise e
    # Since we are on a persistent VM, we save directly to the specified folder path.
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'kia_ai_2b.pth')
    
    print(f"Training complete! Saving weights locally to {save_path}...")
    
    # Save the model weights permanently
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!\n")

def infer_model(model, device, save_dir, tokenizer, prompt="Once upon a time, there was a little"):
    """
    Loads saved model weights and generates output without retraining.
    """
    import os
    save_path = os.path.join(save_dir, 'kia_ai_2b.pth')
    
    if not os.path.exists(save_path):
        print(f"Error: Could not find saved model weights at {save_path}")
        print("Please run with --train first to generate the weights file.")
        return
        
    print(f"Loading weights from {save_path}...")
    try:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    except RuntimeError as e:
        print("\n" + "="*60)
        print(f"❌ ARCHITECTURE MISMATCH ERROR:")
        print(f"You scaled the model up to 1.6-Billion parameters (d_model=2048), but your saved weights in '{save_path}' were trained on a different size!")
        print(f"Please delete the old weights to let the new 1.6B parameter brain train from scratch:")
        print(f"   Run: rm -rf {save_dir}/kia_ai_2b.pth")
        print(f"   Then run the training command again!")
        print("="*60 + "\n")
        return
        
    model.to(device)
    
    # --- GENERATE OUTPUT FOR TESTING ---
    print("\n--- KIA AI IS SPEAKING ---")
    model.eval() # Put model in evaluation mode
    
    # 1. Use the provided prompt
    print(f"Prompt: '{prompt}'")
    
    # 2. Translate English to Math (Token IDs)
    encoded = tokenizer(prompt, return_tensors="pt")
    start_idx = encoded["input_ids"].to(device)
    
    # 3. Ask the math-brain to predict the next 30 tokens
    with torch.no_grad(): # Don't calculate gradients for inference
        generated_tokens = model.generate(start_idx, max_new_tokens=30)
        
    # 4. Translate Math (Token IDs) back to English!
    output_text = tokenizer.decode(generated_tokens[0].tolist())
    
    print("\n--- FINAL OUTPUT ---")
    print(output_text)
    print("--------------------------\n")
    print("It worked! KIA AI is now reading and writing real English!")

def serve_model(model, device, save_dir, tokenizer, ngrok_token=None):
    """
    Starts the FastAPI + Ngrok server to connect to the React app.
    Loads saved model weights.
    """
    import os
    save_path = os.path.join(save_dir, 'kia_ai_2b.pth')
    
    if not os.path.exists(save_path):
        print(f"Error: Could not find saved model weights at {save_path}")
        print("Please run with --train first to generate the weights file.")
        return
        
    print(f"Loading weights from {save_path}...")
    try:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    except RuntimeError as e:
        print("\n" + "="*60)
        print(f"❌ ARCHITECTURE MISMATCH ERROR:")
        print(f"You scaled the model up to 1.6-Billion parameters (d_model=2048), but your saved weights in '{save_path}' were trained on a different size!")
        print(f"Please delete the old weights to let the new 1.6B parameter brain train from scratch before running --serve:\n")
        print(f"   Run: rm -rf {save_dir}/kia_ai_2b.pth")
        print(f"   Then run the training command again!")
        print("="*60 + "\n")
        return
        
    model.to(device)
    model.eval()

    # --- INFERENCE SERVER ---
    # On your VM, make sure you ran: pip install fastapi nest-asyncio pyngrok uvicorn pydantic
    import nest_asyncio
    from pyngrok import ngrok
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    import asyncio
    from pydantic import BaseModel
    
    app = FastAPI(title="KIA AI API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 50
        
    @app.post("/generate")
    async def generate(request: GenerateRequest):
        print(f"Received prompt: {request.prompt}")
        encoded = tokenizer(request.prompt, return_tensors="pt")
        start_idx = encoded["input_ids"].to(device)
        with torch.no_grad():
            generated_tokens = model.generate(start_idx, max_new_tokens=request.max_tokens)
        output_text = tokenizer.decode(generated_tokens[0].tolist())
        return {"response": output_text}

    @app.post("/stream")
    async def stream_generate(request: GenerateRequest):
        print(f"Received streaming prompt: {request.prompt}")
        encoded = tokenizer(request.prompt, return_tensors="pt")
        start_idx = encoded["input_ids"].to(device)
        with torch.no_grad():
            generated_tokens = model.generate(start_idx, max_new_tokens=request.max_tokens)
        output_text = tokenizer.decode(generated_tokens[0].tolist())
        
        async def event_generator():
            words = output_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield chunk
                await asyncio.sleep(0.05)
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    if ngrok_token:
        print("Setting Ngrok Auth Token...")
        ngrok.set_auth_token(ngrok_token)
    else:
        print("WARNING: No --ngrok_token provided. If this is a new VM, Ngrok might fail to connect!")

    ngrok_tunnel = ngrok.connect(8000)
    print("=" * 60)
    print('✅ KIA AI BACKEND IS LIVE!')
    print('🚨 COPY THIS PUBLIC URL TO YOUR REACT APP =>', ngrok_tunnel.public_url)
    print("=" * 60)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

if __name__ == "__main__":
    main()
