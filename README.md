# diffusion-models

Minimal, self-contained diffusion model examples.

## CIFAR-10 Class-Conditional DDPM (Notebook)

Open the notebook `cifar10_ccddpm_minimal.ipynb` for a concise, single-file example that:

- Trains a tiny class-conditional DDPM on CIFAR-10
- Uses classifier-free guidance at sampling time
- Saves a grid of generated samples per class

### Quickstart

1. Install dependencies (Python 3.9+ recommended):
   ```bash
   pip install torch torchvision
   ```

## How this diffusion model works

- **Forward (noising) process**: At each step t, Gaussian noise with variance `beta_t` is added so that after T steps the image is nearly Gaussian. In closed form: `q(x_t | x_0) = N( sqrt(\bar{\alpha}_t) x_0, (1-\bar{\alpha}_t) I )` where `\bar{\alpha}_t = \prod_{s=0}^t (1-\beta_s)`.
- **Objective (epsilon prediction)**: The UNet predicts the added noise `ε`. We train with MSE between the predicted `ε_θ(x_t, t, y)` and the true noise `ε`.
- **Conditioning + classifier-free guidance**: Timestep and class embeddings are summed. During training, labels are randomly dropped (mapped to a dedicated null id) to learn an unconditional branch. At sampling, interpolate: `ε = ε_uncond + s (ε_cond - ε_uncond)`.
- **Reverse process**: Use `ε_θ` to form `\hat{x}_0`, then sample `x_{t-1}` from the closed-form posterior `p_θ(x_{t-1} | x_t)`. Repeat for `t=T-1...0`.
- **Practical tips**: Increase steps/epochs, use EMA weights, cosine schedules, and wider UNets for better quality.