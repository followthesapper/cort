"""
CoRT Model: Complete Coherence-Routed Transformer for language modeling.

This is the main model class that combines all CoRT components into
a ready-to-use transformer for language modeling tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..core.config import CoRTConfig
from ..layers.cort_layer import CoRTLayer


class CoRTModel(nn.Module):
    """
    Coherence-Routed Transformer for language modeling.

    A transformer that uses phase coherence metrics to intelligently
    route tokens between full attention and lightweight mixing,
    achieving better perplexity with comparable computational cost.

    Args:
        config: CoRTConfig with model hyperparameters

    Example:
        >>> config = CoRTConfig(vocab_size=50257, d_model=512, n_layers=6)
        >>> model = CoRTModel(config)
        >>> input_ids = torch.randint(0, 50257, (2, 128))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 128, 50257])
    """

    def __init__(self, config: CoRTConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            CoRTLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                route_frac=config.route_frac,
                adaptive_routing=config.adaptive_routing,
                coherence_mode=config.coherence_mode,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection (language model head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.

        Args:
            input_ids: [batch, seq_len] token indices
            attention_mask: Optional [batch, seq_len] attention mask
            return_stats: Return layer-wise routing statistics

        Returns:
            logits: [batch, seq_len, vocab_size] output logits
        """
        B, L = input_ids.shape

        # Token + position embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.dropout(hidden_states)

        # Collect stats if requested
        all_stats = [] if return_stats else None

        # Apply transformer layers
        for layer in self.layers:
            hidden_states, stats = layer(hidden_states, attention_mask, return_stats)
            if return_stats:
                all_stats.append(stats)

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        if return_stats:
            return logits, all_stats

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: [batch, seq_len] starting tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 to disable)
            top_p: Nucleus sampling threshold
            eos_token_id: Stop generation at this token

        Returns:
            generated: [batch, seq_len + new_tokens] full sequence
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate to max_seq_len
                context = input_ids[:, -self.max_seq_len:]

                # Get logits
                logits = self(context)
                next_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
                    next_logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return input_ids

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "CoRTModel":
        """
        Load a pretrained model from disk.

        Args:
            path: Path to saved model (without extension)
            device: Device to load model on

        Returns:
            Loaded CoRTModel
        """
        import json

        # Load config
        with open(f"{path}_config.json", "r") as f:
            config_dict = json.load(f)
        config = CoRTConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(f"{path}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        return model.to(device)

    def save_pretrained(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save model (without extension)
        """
        import json

        # Save config
        with open(f"{path}_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        torch.save(self.state_dict(), f"{path}.pt")

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
