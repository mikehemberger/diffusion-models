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