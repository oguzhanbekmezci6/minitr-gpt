# MiniTR-GPT

<p align="center">
  <img src="assets/minitr-gpt-cover.png" alt="MiniTR-GPT architecture" width="1000">
</p>

<p align="center">
  <strong>A compact Turkish GPT-style language model built from scratch with PyTorch</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-orange" alt="Decoder-Only Transformer">
  <img src="https://img.shields.io/badge/Language-Turkish-red" alt="Turkish">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License">
</p>

---

## Overview

**MiniTR-GPT** is a compact, decoder-only Transformer language model developed from first principles with PyTorch.

The purpose of this project is not to compete with large-scale language models. Its goal is to make the internal mechanics of GPT-style architectures understandable, reproducible, and easy to experiment with.

Instead of relying on high-level modules such as `torch.nn.Transformer` or `torch.nn.MultiheadAttention`, the project implements the core building blocks directly:

- Character-level tokenization
- Token embeddings
- Positional embeddings
- Query, key, and value projections
- Masked multi-head self-attention
- Causal masking
- Feed-forward neural networks
- Residual connections
- Layer normalization
- Dropout
- Autoregressive text generation
- Training and validation loops
- Checkpoint saving and loading

The model is trained on Turkish text and generates new text one character at a time.

---

## Project Goals

This project was created to:

- Understand how GPT-style language models work internally
- Implement self-attention without high-level Transformer abstractions
- Explore causal language modeling on Turkish text
- Build a complete training pipeline from scratch
- Experiment with model size, context length, and generation settings
- Create an educational foundation for larger NLP and LLM projects

---

## Architecture

MiniTR-GPT follows a decoder-only Transformer architecture.

```text
Input Text
   │
   ▼
Character-Level Tokenizer
   │
   ▼
Token Embeddings + Positional Embeddings
   │
   ▼
┌──────────────────────────────────────────────┐
│ Transformer Block × N                       │
│                                              │
│ LayerNorm                                    │
│    │                                         │
│ Masked Multi-Head Self-Attention             │
│    │                                         │
│ Residual Connection                          │
│    │                                         │
│ LayerNorm                                    │
│    │                                         │
│ Feed-Forward Neural Network                  │
│    │                                         │
│ Residual Connection                          │
└──────────────────────────────────────────────┘
   │
   ▼
Final LayerNorm
   │
   ▼
Linear Language Modeling Head
   │
   ▼
Next-Character Probabilities
```

### Self-Attention

For each attention head, the input representation is projected into query, key, and value matrices:

```text
Q = XWq
K = XWk
V = XWv
```

Scaled dot-product attention is then computed as:

```text
Attention(Q, K, V) = softmax((QKᵀ / √dk) + causal_mask)V
```

The causal mask prevents each token from attending to future tokens.

---

## Example Demo Configuration

A small CPU-friendly demonstration configuration may contain:

| Parameter | Example Value |
|---|---:|
| Vocabulary size | 65 characters |
| Training tokens | 9,141 |
| Validation tokens | 1,016 |
| Transformer layers | 2 |
| Attention heads | 2 |
| Embedding dimension | 64 |
| Context length | 64 |
| Batch size | 16 |
| Model parameters | Approximately 104,256 |
| Training iterations | 300 |

These values are intended for educational demonstrations. Larger datasets and model configurations are required for more coherent text generation.

---

## Repository Structure

```text
MiniTR-GPT/
│
├── assets/
│   └── minitr-gpt-cover.png
│
├── checkpoints/
│   └── demo/
│
├── data/
│   └── sample_turkish.txt
│
├── minitr_gpt/
│   ├── __init__.py
│   ├── model.py
│   ├── tokenizer.py
│   └── utils.py
│
├── demo.py
├── generate.py
├── train.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The exact structure may differ depending on the current version of the project.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/oguzhanbekmezci6/MiniTR-GPT.git
cd MiniTR-GPT
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Activate the environment on macOS or Linux:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train the model with the included sample Turkish dataset:

```bash
python train.py \
  --data data/sample_turkish.txt \
  --out-dir checkpoints/demo \
  --max-iters 300 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 16 \
  --block-size 64 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64
```

Windows PowerShell:

```powershell
python train.py `
  --data data/sample_turkish.txt `
  --out-dir checkpoints/demo `
  --max-iters 300 `
  --eval-interval 50 `
  --eval-iters 10 `
  --batch-size 16 `
  --block-size 64 `
  --n-layer 2 `
  --n-head 2 `
  --n-embd 64
```

A shorter one-line command:

```bash
python train.py --data data/sample_turkish.txt --out-dir checkpoints/demo --max-iters 300 --eval-interval 50 --eval-iters 10 --batch-size 16 --block-size 64 --n-layer 2 --n-head 2 --n-embd 64
```

---

## Text Generation

Generate text from a trained checkpoint:

```bash
python generate.py \
  --checkpoint checkpoints/demo \
  --prompt "Türkiye" \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

Windows PowerShell:

```powershell
python generate.py `
  --checkpoint checkpoints/demo `
  --prompt "Türkiye" `
  --max-new-tokens 300 `
  --temperature 0.8 `
  --top-k 40
```

Generation quality depends heavily on:

- Dataset size
- Dataset quality
- Training duration
- Model capacity
- Context length
- Temperature
- Top-k sampling

---

## Running the Demo

The demo script can be used to train a small model and test text generation:

```bash
python demo.py
```

A typical training log may look like:

```text
Device: cpu
Vocabulary: 65 characters
Training tokens: 9,141
Validation tokens: 1,016
Model parameters: 104,256

Step 0:
train loss = 4.2070
validation loss = ...
```

The exact values vary depending on the dataset, configuration, random seed, and software version.

---

## Core Components

### Character-Level Tokenizer

The tokenizer creates a vocabulary from the unique characters in the training corpus.

```python
chars = sorted(list(set(text)))
stoi = {character: index for index, character in enumerate(chars)}
itos = {index: character for character, index in stoi.items()}
```

Encoding converts text into integer token IDs:

```python
def encode(text: str) -> list[int]:
    return [stoi[character] for character in text]
```

Decoding converts token IDs back into text:

```python
def decode(tokens: list[int]) -> str:
    return "".join(itos[token] for token in tokens)
```

### Causal Self-Attention

Each token can attend only to itself and previous tokens.

```python
attention_scores = query @ key.transpose(-2, -1)
attention_scores = attention_scores / (key.size(-1) ** 0.5)
attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))
attention_weights = torch.softmax(attention_scores, dim=-1)
output = attention_weights @ value
```

### Feed-Forward Network

Each Transformer block contains a position-wise neural network:

```python
nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.GELU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout),
)
```

### Autoregressive Generation

The model repeatedly predicts and samples the next token:

```python
for _ in range(max_new_tokens):
    context = token_ids[:, -block_size:]
    logits, _ = model(context)
    logits = logits[:, -1, :] / temperature
    probabilities = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    token_ids = torch.cat((token_ids, next_token), dim=1)
```

---

## Hyperparameters

| Argument | Description |
|---|---|
| `--data` | Path to the training text file |
| `--out-dir` | Directory used for checkpoints and outputs |
| `--batch-size` | Number of sequences per training batch |
| `--block-size` | Maximum context length |
| `--max-iters` | Total number of training iterations |
| `--eval-interval` | Number of steps between evaluations |
| `--eval-iters` | Number of batches used for evaluation |
| `--learning-rate` | Optimizer learning rate |
| `--n-layer` | Number of Transformer blocks |
| `--n-head` | Number of self-attention heads |
| `--n-embd` | Embedding dimension |
| `--dropout` | Dropout probability |
| `--seed` | Random seed for reproducibility |
| `--device` | Training device such as CPU or CUDA |

---

## Checkpoints and Outputs

Training outputs are stored in the directory specified by `--out-dir`.

Example:

```text
checkpoints/demo/
├── model.pt
├── config.json
├── tokenizer.json
├── training_history.json
└── generated_sample.txt
```

Depending on the project version, file names may differ.

A checkpoint should normally contain:

- Model parameters
- Model configuration
- Tokenizer vocabulary
- Optimizer state
- Current training step
- Training and validation losses

---

## Technologies

- Python
- PyTorch
- NumPy
- tqdm
- Transformer architecture
- Natural language processing
- Character-level language modeling
- Autoregressive generation

---

## Current Limitations

- Character-level tokenization is less efficient than subword tokenization.
- The included sample dataset is too small for high-quality language generation.
- The model is intended for educational use rather than production deployment.
- Training on CPU can be slow for larger configurations.
- Generated text may contain repetition, spelling errors, or incoherent sequences.
- The project does not include instruction tuning, RLHF, or safety alignment.
- The model does not have access to current information or external tools.

---

## Future Improvements

- Add Byte Pair Encoding or SentencePiece tokenization
- Train on a larger and cleaner Turkish corpus
- Add mixed-precision GPU training
- Add gradient accumulation
- Add learning-rate warmup and cosine decay
- Add validation perplexity reporting
- Add TensorBoard or Weights & Biases integration
- Support checkpoint resume
- Add beam search and nucleus sampling
- Compare pre-norm and post-norm Transformer blocks
- Add model export and inference API
- Build a Streamlit or Gradio demonstration interface

---

## Educational Value

MiniTR-GPT demonstrates that a GPT-style model is not a single black-box component. It is a structured system built from understandable mathematical and software components.

The project provides hands-on experience with:

- Language model training
- Tensor dimensions
- Attention mechanics
- Masking
- Gradient-based optimisation
- Overfitting and validation
- Sampling strategies
- Reproducible machine learning workflows

---

## Disclaimer

MiniTR-GPT is an educational language model.

It is not intended for:

- High-stakes decision-making
- Production use without additional testing
- Generating verified factual information
- Replacing large-scale commercial language models

Generated content may be inaccurate, incomplete, or nonsensical.

---

## Author

**Oğuzhan Bekmezci**

Statistics Graduate  
Data Science · Machine Learning · Artificial Intelligence

[GitHub Profile](https://github.com/oguzhanbekmezci6)

---

## License

This project is licensed under the MIT License.
