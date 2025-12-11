#!/usr/bin/env python3
"""
Example: Train a CoRT language model on WikiText-103.

This script demonstrates the full training pipeline for CoRT,
including data loading, model creation, training, and generation.

Usage:
    python examples/train_language_model.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cort import CoRTModel, CoRTConfig


def load_data(max_sequences: int = 10000, seq_len: int = 256):
    """Load and tokenize WikiText-103."""
    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError:
        print("Please install training dependencies:")
        print("  pip install cort-transformer[training]")
        return None, None

    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Concatenate text
    text = "\n".join([ex["text"] for ex in dataset if ex["text"].strip()])

    # Tokenize
    print("Tokenizing...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    print(f"  Total tokens: {len(tokens):,}")

    # Create sequences
    n_sequences = min(max_sequences, len(tokens) // seq_len)
    data = torch.tensor(tokens[: n_sequences * seq_len], dtype=torch.long)
    data = data.view(n_sequences, seq_len)
    print(f"  Training sequences: {n_sequences:,}")

    return data, enc


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch[0].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    data, enc = load_data(max_sequences=10000, seq_len=256)
    if data is None:
        return

    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True)

    # Create model
    print("\nCreating CoRT model...")
    config = CoRTConfig(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=256,
        adaptive_routing=True,
    )
    model = CoRTModel(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training
    print("\nTraining...")
    for epoch in range(3):
        loss = train_epoch(model, loader, optimizer, device)
        ppl = torch.exp(torch.tensor(loss)).item()
        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, PPL={ppl:.2f}")

    # Save model
    print("\nSaving model...")
    model.save_pretrained("cort_wikitext")

    # Test generation
    print("\nGenerating text...")
    prompt = "The history of artificial intelligence"
    input_ids = torch.tensor([enc.encode(prompt)], device=device)

    generated = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {enc.decode(generated[0].tolist())}")


if __name__ == "__main__":
    main()
