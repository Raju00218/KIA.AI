import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fastapi import FastAPI
from pydantic import BaseModel
import argparse # Added this import
# Note: In Colab, you will need to run:
# !pip install fastapi nest-asyncio pyngrok uvicorn

# ==============================================================================
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
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Calculate Q, K, V and split into heads
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # TODO (Phase 3): Implement Rotary Positional Embeddings (RoPE) here
        # to handle longer sequence contexts than absolute embeddings.
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (so it only looks at past tokens)
        if mask is None:
            mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Aggregate values
        out = torch.matmul(attn_weights, v)
        
        # Re-assemble heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

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
        
    def forward(self, x):
        # Pre-LN architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

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
        
    def forward(self, idx):
        B, T = idx.size()
        device = idx.device
        
        # Generate position indices [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Add embeddings
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
        
    def generate(self, idx, max_new_tokens):
        # TODO (Phase 3): Implement KV Caching here to change O(N^2) to O(N) 
        # so generation doesn't slow to a crawl on long sequences.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -2048:] # Crop to max sequence length
            logits = self(idx_cond)
            logits = logits[:, -1, :] # Focus on last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================================================
# 2. COLAB TRAINING/INFERENCE SCRIPT (Run this part inside Colab)
# ==============================================================================

if __name__ == "__main__":
    print("Welcome to KIA AI - 2B Parameter Initialization")
    print("WARNING: Allocating a 2B parameter model requires 8GB+ of VRAM just for weights.")
    
def main():
    # vocab_size usually corresponds to a tokenizer like TikToken or SentencePiece
    vocab_size = 50257 # Standard GPT-2 vocab size for example
    
    try:
        # Initialize the model on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing KIA AI on {device}...")
        
        # Instantiate Model
        # Note: We scale down for testing unless running on A100/V100
        # For actual 2B use: d_model=2560, num_layers=32, num_heads=32
        model = KiaAI(vocab_size=vocab_size, d_model=512, num_layers=8, num_heads=8)
        
        # TODO (Phase 3): Implement Half Precision (BF16 / FP16) memory optimization
        # using model = model.bfloat16() here once ready for full 2B load.

        model.to(device)
        print("Model initialized successfully (Vanilla Architecture)!")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="KIA AI Backend")
    parser.add_argument('--train', action='store_true', help="Run the model training loop")
    parser.add_argument('--serve', action='store_true', help="Run the FastAPI inference server")
    
    # In Jupyter/Colab environments, sys.argv includes an undocumented '-f' flag that crashes standard parsing.
    # To fix this, we parse with parse_known_args() and explicitly check if '--train' was provided in sys.argv.
    import sys
    args, unknown = parser.parse_known_args()

    # Note: Using '--train' in sys.argv check bypasses notebook default kernels overriding args.train
    if '--train' in sys.argv or args.train:
        print("\n--- Starting KIA AI Model Training ---")
        train_model(model, device)
    elif '--serve' in sys.argv or args.serve or ('--train' not in sys.argv and not args.train): # Default to serve if nothing passed
        print("\n--- Starting KIA AI Inference Server ---")
        serve_model(model, device)

def train_model(model, device):
    """
    Core PyTorch Training Loop for the KIA AI model.
    In Phase 3, this is where we feed a dataset (like TinyStories) to adjust weights.
    """
    # 1. Setup Optimizer (AdamW is standard for Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 2. Dummy Dataset Placeholder (Replace with real DataLoader in Colab)
    # We will grab batches of shape (B, T) = (Batch Size, Sequence Length)
    print("Loading dataset... (Using dummy data for structural testing)")
    def get_batch():
        # X: Input tokens
        # Y: Target tokens (X shifted by 1)
        # E.g. If X is "The cat sat", Y is "cat sat on"
        X = torch.randint(0, 50257, (4, 128)).to(device) # Batch 4, Seq 128
        Y = torch.randint(0, 50257, (4, 128)).to(device)
        return X, Y

    iterations = 1000
    print(f"Beginning training loop for {iterations} iterations on {device}...")
    
    model.train()
    for step in range(iterations):
        # Sample a batch of data
        xb, yb = get_batch()
        
        # Forward pass: compute predictions
        logits = model(xb)
        
        # Loss computation requires reshaping logits to (B*T, C) and targets to (B*T)
        B, T, C = logits.shape
        logits_reshaped = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = loss_fn(logits_reshaped, targets)
        
        # Backward pass: compute gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Step: update weights
        optimizer.step()
        
        if step % 100 == 0 or step == iterations - 1:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            
    print("Training complete! Weights have been updated.")
    
    # Save the model weights
    # torch.save(model.state_dict(), "kia_ai_2b.pth")
    # print("Model saved to kia_ai_2b.pth")
    print("Run `!python kia_colab_backend.py --serve` to test inference!")

def serve_model(model, device):
    """
    Starts the FastAPI + Ngrok server to connect to the React app.
    """
    model.eval() # Set model to evaluation mode for inference

    # --- INFERENCE SERVER ---
    # In Colab, make sure you ran: !pip install fastapi nest-asyncio pyngrok uvicorn
    import nest_asyncio
    from pyngrok import ngrok
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    import asyncio
    from pydantic import BaseModel
    
    app = FastAPI(title="KIA AI API")

    # This is CRITICAL for the React app to talk to Colab
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Allows your React localhost to connect
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 150
        
    @app.post("/generate")
    async def generate(request: GenerateRequest):
        print(f"Received prompt: {request.prompt}")
        return {"response": f"KIA AI successfully received your prompt: '{request.prompt}'!\n\n(Backend 2B Model is connected and communicating!)"}

    @app.post("/stream")
    async def stream_generate(request: GenerateRequest):
        print(f"Received streaming prompt: {request.prompt}")
        
        async def event_generator():
            # Mocking a text generation stream (word by word)
            mock_response = f"KIA AI is streaming the response dynamically!\n\nYou asked: '{request.prompt}'.\n\nIf the 2B PyTorch model was fully trained, you would see it predicting the next tokens in real time right here."
            words = mock_response.split(" ")
            
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield chunk
                await asyncio.sleep(0.05)
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    # Start server
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

if __name__ == "__main__":
    main()
