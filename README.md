# GaLore

**Overview**:

The `GaLore` optimizer extends the Adam framework by incorporating low-rank projection techniques into the gradient update process. For parameters with two or more dimensions, it leverages a dedicated projector (via `GaLoreProjector`) to project the gradient into a lower-dimensional subspace before applying the adaptive update. This extra projection step is intended to better capture and exploit the low-rank structure of weight matrices, potentially leading to more robust convergence. Along with standard exponential moving averages for gradients and their squares, GaLore supports decoupled weight decay and bias correction via learning rate scaling.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The base step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied in a decoupled manner if enabled.
- **`rank`** *(int, optional)*: The target rank for low-rank projection. When provided and the parameter is multi-dimensional, the projector is activated.
- **`update_proj_gap`** *(int, optional)*: Frequency gap for updating the projection; governs how often the low-rank projection is recalculated.
- **`scale`** *(float, optional)*: Scaling factor applied within the projector.
- **`projection_type`** *(str, optional)*: Specifies the type of projection to perform (e.g., symmetric or asymmetric).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients.
- **`name`** *(str, default="galore")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
from galore import GaLore

# Instantiate the GaLore optimizer with low-rank projection enabled
optimizer = GaLore(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-6,
    weight_decay=1e-4,
    rank=10,  # Enable low-rank projection for parameters with rank ≥ 2
    update_proj_gap=50,
    scale=0.5,
    projection_type="symmetric"
)
