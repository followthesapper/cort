"""
CoRT Training Utilities: Simple training loop and configuration.

Provides a lightweight training interface for CoRT models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Iterator
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 100
    max_steps: Optional[int] = None

    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000

    # Checkpointing
    save_dir: str = "checkpoints"


class Trainer:
    """
    Simple training loop for CoRT models.

    Example:
        >>> config = TrainingConfig(learning_rate=1e-4)
        >>> trainer = Trainer(model, config)
        >>> trainer.train(train_loader, num_epochs=3)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # State
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")

    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()

        # Move to device
        batch = batch.to(self.device)

        # Forward pass
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = self.model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()
        self.step += 1

        return loss.item()

    @torch.no_grad()
    def evaluate(self, eval_loader: Iterator) -> float:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in eval_loader:
            batch = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: Iterator,
        num_epochs: int = 1,
        eval_loader: Optional[Iterator] = None,
        log_fn: Optional[Callable] = None,
    ):
        """
        Full training loop.

        Args:
            train_loader: Iterator yielding batches
            num_epochs: Number of epochs to train
            eval_loader: Optional validation data
            log_fn: Logging function (receives dict of metrics)
        """
        if log_fn is None:
            log_fn = lambda x: print(x)

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0
            num_batches = 0

            for batch in train_loader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    ppl = torch.exp(torch.tensor(avg_loss)).item()
                    log_fn({
                        "step": self.step,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "ppl": ppl,
                    })

                # Evaluation
                if eval_loader and self.step % self.config.eval_interval == 0:
                    eval_loss = self.evaluate(eval_loader)
                    eval_ppl = torch.exp(torch.tensor(eval_loss)).item()
                    log_fn({
                        "step": self.step,
                        "eval_loss": eval_loss,
                        "eval_ppl": eval_ppl,
                    })

                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best")

                # Checkpointing
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.step}")

                # Max steps
                if self.config.max_steps and self.step >= self.config.max_steps:
                    return

            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            log_fn({
                "epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss,
                "epoch_ppl": torch.exp(torch.tensor(avg_epoch_loss)).item(),
            })

            self.save_checkpoint(f"epoch_{epoch + 1}")

    def save_checkpoint(self, name: str):
        """Save a checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        path = save_dir / name
        self.model.save_pretrained(str(path))

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        self.model = self.model.from_pretrained(path, self.device)
