# CoRT API Guide

Complete API reference for the Coherence-Routed Transformer.

## Table of Contents

- [Core Module](#core-module)
- [Layers Module](#layers-module)
- [Models Module](#models-module)
- [Utils Module](#utils-module)

---

## Core Module

### `cort.core.coherence`

Phase coherence computation functions.

#### `r_bar_from_phases`

```python
def r_bar_from_phases(phases: torch.Tensor, dim: int = -1) -> torch.Tensor
```

Compute Mean Resultant Length (R-bar) from phase angles.

**Parameters:**
- `phases`: Phase angles in radians
- `dim`: Dimension along which to compute R-bar

**Returns:**
- R-bar values in [0, 1]

**Example:**
```python
phases = torch.tensor([0.1, 0.1, 0.1, 0.1])  # Aligned
r_bar = r_bar_from_phases(phases)
# tensor(0.9950)
```

---

#### `compute_phase_coherence`

```python
def compute_phase_coherence(
    hidden_states: torch.Tensor,
    mode: str = "token",
) -> torch.Tensor
```

Compute phase coherence from hidden states.

**Parameters:**
- `hidden_states`: Token embeddings [batch, seq_len, d_model]
- `mode`: "token" (per-token) or "sequence" (per-sequence)

**Returns:**
- Coherence scores

**Example:**
```python
hidden = torch.randn(2, 128, 512)
coherence = compute_phase_coherence(hidden, mode="token")
# Shape: [2, 128]
```

---

#### `compute_local_coherence`

```python
def compute_local_coherence(
    hidden_states: torch.Tensor,
    window: int = 8,
) -> torch.Tensor
```

Compute local phase coherence using sliding windows.

**Parameters:**
- `hidden_states`: Token embeddings [batch, seq_len, d_model]
- `window`: Size of the local window

**Returns:**
- Local coherence scores [batch, seq_len]

---

### `cort.core.config`

#### `CoRTConfig`

```python
@dataclass
class CoRTConfig:
    # Architecture
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: Optional[int] = None
    max_seq_len: int = 1024
    dropout: float = 0.1

    # Routing
    route_frac: float = 0.15
    r_bar_threshold: float = 0.5
    adaptive_routing: bool = True
    coherence_window: int = 8
    coherence_mode: str = "combined"

    # Training
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5
```

Model configuration dataclass.

**Methods:**
- `to_dict()`: Convert to dictionary
- `from_dict(d)`: Create from dictionary
- `small()`: Small model preset (~48M params)
- `medium()`: Medium model preset (~124M params)
- `large()`: Large model preset (~350M params)

**Example:**
```python
# Custom config
config = CoRTConfig(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
)

# Preset
config = CoRTConfig.medium()
```

---

## Layers Module

### `cort.layers.router`

#### `CoherentRouter`

```python
class CoherentRouter(nn.Module):
    def __init__(self, config: RoutingConfig, d_model: int)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Routes tokens based on phase coherence.

**Returns:**
- `attn_mask`: Boolean mask for attention tokens
- `mix_mask`: Boolean mask for mixing tokens
- `routing_scores`: Per-token routing scores

---

#### `AdaptiveCoherentRouter`

```python
class AdaptiveCoherentRouter(CoherentRouter):
    def __init__(
        self,
        config: RoutingConfig,
        d_model: int,
        target_coherence: float = 0.5,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
    )
```

Router with PID-tuned adaptive routing fraction.

---

### `cort.layers.mixer`

#### `CoherenceMixer`

```python
class CoherenceMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        mix_strength: float = 0.5,
        learnable: bool = True,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        coherence_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor
```

Phase-aware token mixing for high-coherence tokens.

---

### `cort.layers.attention`

#### `MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

Standard multi-head self-attention.

---

### `cort.layers.cort_layer`

#### `CoRTLayer`

```python
class CoRTLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        route_frac: float = 0.15,
        adaptive_routing: bool = True,
        coherence_mode: str = "combined",
        layer_norm_eps: float = 1e-5,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]
```

Complete CoRT transformer layer with routing.

**Returns:**
- `output`: Transformed hidden states
- `stats`: Optional dict with routing statistics

---

## Models Module

### `cort.models.transformer`

#### `CoRTModel`

```python
class CoRTModel(nn.Module):
    def __init__(self, config: CoRTConfig)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> torch.Tensor

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "CoRTModel"

    def save_pretrained(self, path: str)

    def num_parameters(self, trainable_only: bool = True) -> int
```

Complete CoRT model for language modeling.

**Example:**
```python
config = CoRTConfig.small()
model = CoRTModel(config)

# Forward pass
logits = model(input_ids)

# Generation
generated = model.generate(prompt_ids, max_new_tokens=100)

# Save/load
model.save_pretrained("my_model")
model = CoRTModel.from_pretrained("my_model")
```

---

## Utils Module

### `cort.utils.training`

#### `TrainingConfig`

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    save_dir: str = "checkpoints"
```

Training configuration.

---

#### `Trainer`

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
    )

    def train(
        self,
        train_loader: Iterator,
        num_epochs: int = 1,
        eval_loader: Optional[Iterator] = None,
        log_fn: Optional[Callable] = None,
    )

    def train_step(self, batch: torch.Tensor) -> float

    def evaluate(self, eval_loader: Iterator) -> float

    def save_checkpoint(self, name: str)

    def load_checkpoint(self, path: str)
```

Simple training loop for CoRT models.

**Example:**
```python
trainer = Trainer(model, TrainingConfig())
trainer.train(train_loader, num_epochs=3)
```
