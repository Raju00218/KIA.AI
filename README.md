# Kia.ai: 1.6 Billion Parameter Generative AI Built From Scratch

Kia.ai is a custom-built, 1.6 Billion parameter Large Language Model (LLM) utilizing a PyTorch Transformer architecture. Unlike wrappers that simply ping external APIs (like OpenAI), this project represents a complete, ground-up implementation of a Generative AI brain, including the mathematical architecture, datasets, industrial training loops, and a React frontend.

## The Architecture (The Brain)
Built entirely in PyTorch, the underlying model is a scaled-up implementation of the standard Decoder-Only Transformer architecture.

**Key Technical Specifications:**
*   **Parameters:** ~1.6 Billion
*   **Dimensionality (`d_model`):** 2048
*   **Layers:** 32
*   **Attention Heads:** 32
*   **Optimization:** `bfloat16` half-precision to fit the massive matrix weights within 24GB of VRAM.
*   **Inference:** Custom KV Caching implemented to transform generation complexity from O(N²) to O(N), allowing for rapid token streaming.

## Industrial Training Mechanics
Training a 1.6B parameter model from scratch requires advanced techniques to prevent catastrophic forgetting and gradient explosion. This project utilizes a Two-Stage training process:

### Stage 1: Pre-training (Current Phase)
To teach the AI fundamental English grammar, the model is trained exclusively on highly-curated, clean datasets (like `TinyStories`).
*   **Data Curation:** A strict Regex pipeline strips all non-ASCII characters and formatting noise *before* tokenization.
*   **Cosine Annealing Learning Rate:** A 3-stage LR scheduler (Warmup -> Core Learning -> Cooldown) ensures the weights adjust safely without destroying previous knowledge.
*   **Overfitting Protection:** Gradient clipping (max norm = 1.0) and an automated loss monitor that reverts to the best checkpoint if the model begins memorizing noise (generating gibberish).

## The Tech Stack (Deployment)
The ecosystem is split into two halves: the heavy computing backend on Google Cloud, and the lightweight frontend on the client.

*   **Backend (Google Cloud VM):** The `kia_vm_backend.py` script runs on a persistent VM with an L4 GPU. It handles the PyTorch training loop and hosts a `FastAPI` server for inference.
*   **Tunneling:** `Ngrok` securely exposes the FastAPI server to the internet.
*   **Frontend (React):** A custom, beautifully styled React UI that communicates with the VM via Server-Sent Events (SSE) to display generated tokens in real-time.

## Setup & Usage (Local Development)

### 1. React Frontend
To run the user interface locally:
```bash
npm install
npm run dev
```

### 2. The AI Backend (Google Cloud VM)
The model weights (`kia_ai_2b.pth`) are too large for GitHub (3+ GB) and reside only on the VM.

**To Train the Model:**
```bash
python model_folder/kia_vm_backend.py --train
```

**To Serve the API:**
```bash
python model_folder/kia_vm_backend.py --serve --ngrok_token YOUR_TOKEN
```
*(Once serving, paste the generated Ngrok URL into the `src/services/apiService.js` file in the React frontend to connect them).*

---
*Created by Raju Allaveni for the Generative AI Spotlight Event.*
