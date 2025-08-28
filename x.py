```python
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Diffusion schedule and precomputations
T = 1000

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=DEVICE)

betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), alphas_cumprod[:-1]], dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

def extract(a: torch.Tensor, t: torch.LongTensor, shape: Tuple[int, ...]):
    out = a.gather(-1, t)
    while len(out.shape) < len(shape):
        out = out.unsqueeze(-1)
    return out

# Model: small class-conditional UNet for 32x32
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.LongTensor):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / float(half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond(cond).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor):
        return self.op(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    def forward(self, x: torch.Tensor):
        return self.conv(self.up(x))

class UNet32(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, class_embed_num: int = 11):
        super().__init__()
        ch1, ch2, ch3 = base_channels, base_channels * 2, base_channels * 2
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(class_embed_num, time_dim)

        self.in_conv = nn.Conv2d(in_channels, ch1, 3, padding=1)
        self.rb1 = ResidualBlock(ch1, ch1, time_dim)
        self.rb2 = ResidualBlock(ch1, ch1, time_dim)
        self.down1 = Down(ch1, ch2)
        self.rb3 = ResidualBlock(ch2, ch2, time_dim)
        self.down2 = Down(ch2, ch3)

        self.mid1 = ResidualBlock(ch3, ch3, time_dim)
        self.mid2 = ResidualBlock(ch3, ch3, time_dim)

        self.up1 = Up(ch3, ch2)
        self.rb4 = ResidualBlock(ch2 + ch2, ch2, time_dim)
        self.up2 = Up(ch2, ch1)
        self.rb5 = ResidualBlock(ch1 + ch1, ch1, time_dim)

        self.out_norm = nn.GroupNorm(8, ch1)
        self.out_conv = nn.Conv2d(ch1, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.LongTensor, y: torch.LongTensor):
        temb = self.time_mlp(t)
        cemb = self.class_emb(y)
        cond = temb + cemb

        x0 = self.in_conv(x)
        s1 = self.rb2(self.rb1(x0, cond), cond)
        d1 = self.down1(s1)
        s2 = self.rb3(d1, cond)
        d2 = self.down2(s2)

        m = self.mid2(self.mid1(d2, cond), cond)

        u1 = self.up1(m)
        u1 = torch.cat([u1, s2], dim=1)
        u1 = self.rb4(u1, cond)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, s1], dim=1)
        u2 = self.rb5(u2, cond)

        out = self.out_conv(F.silu(self.out_norm(u2)))
        return out

# Diffusion utilities
def q_sample(x0: torch.Tensor, t: torch.LongTensor, noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(x0)
    return extract(sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

@torch.no_grad()
def p_mean_variance(model: nn.Module, x_t: torch.Tensor, t: torch.LongTensor, y: torch.LongTensor, guidance_scale: float = 0.0):
    null_y = torch.full_like(y, 10)
    eps_cond = model(x_t, t, y)
    if guidance_scale > 0:
        eps_uncond = model(x_t, t, null_y)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    else:
        eps = eps_cond
    x0_pred = (x_t - extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps) / extract(sqrt_alphas_cumprod, t, x_t.shape)
    x0_pred = x0_pred.clamp(-1.0, 1.0)
    mean = extract(posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(posterior_mean_coef2, t, x_t.shape) * x_t
    log_var = extract(posterior_log_variance_clipped, t, x_t.shape)
    return mean, log_var

@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: torch.LongTensor, y: torch.LongTensor, guidance_scale: float = 0.0):
    mean, log_var = p_mean_variance(model, x_t, t, y, guidance_scale)
    if (t == 0).all():
        return mean
    noise = torch.randn_like(x_t)
    return mean + (0.5 * log_var).exp() * noise

@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: Tuple[int, int, int, int], steps: int, labels: torch.LongTensor, guidance_scale: float = 0.0):
    b = shape[0]
    x = torch.randn(shape, device=DEVICE)
    steps = int(steps)
    timesteps = torch.linspace(T - 1, 0, steps, device=DEVICE).long()
    for t_scalar in timesteps:
        t = torch.full((b,), int(t_scalar.item()), device=DEVICE, dtype=torch.long)
        x = p_sample(model, x, t, labels, guidance_scale)
    return x

def main():
    # Config
    EPOCHS = 1
    MAX_STEPS = 200
    BATCH_SIZE = 128 if torch.cuda.is_available() else 32
    LEARNING_RATE = 2e-4
    GUIDANCE_SCALE = 3.0
    UNCOND_PROB = 0.1
    SAMPLE_STEPS = 250
    SAVE_DIR = "outputs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Model and optimizer
    model = UNet32(in_channels=3, base_channels=64, class_embed_num=11).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train
    global_step = 0
    model.train()
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            t = torch.randint(0, betas.shape[0], (images.size(0),), device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(images)
            x_t = q_sample(images, t, noise)

            # classifier-free guidance training: randomly drop labels (use null id=10)
            drop_mask = (torch.rand_like(labels.float()) < UNCOND_PROB)
            labels_train = labels.clone()
            labels_train[drop_mask] = 10

            eps_pred = model(x_t, t, labels_train)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}: loss={loss.item():.4f}")
            if global_step >= MAX_STEPS:
                break
        if global_step >= MAX_STEPS:
            break

    # Sample per class
    model.eval()
    with torch.no_grad():
        n_per_class = 8
        labels = torch.arange(0, 10, device=DEVICE).repeat_interleave(n_per_class)
        samples = p_sample_loop(model, (labels.size(0), 3, 32, 32), steps=SAMPLE_STEPS, labels=labels, guidance_scale=GUIDANCE_SCALE)
        grid = vutils.make_grid((samples.clamp(-1, 1) + 1) / 2, nrow=n_per_class)
        out_path = os.path.join(SAVE_DIR, "cifar10_ccddpm_samples.png")
        vutils.save_image(grid, out_path)
        print("Saved samples to", out_path)

if __name__ == "__main__":
    print("Device:", DEVICE)
    main()