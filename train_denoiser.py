"""
Train the EDM denoiser on Fashion-MNIST using denoising score matching.

Loss:  E_σ E_x E_n [ w(σ) · ||D(x+σn; σ) − x||² ]
where  w(σ) = (σ² + σ_data²) / (σ · σ_data)²
and    ln(σ) ~ N(P_mean, P_std²)

Usage:
  python train_denoiser.py                      # defaults (5 epochs)
  python train_denoiser.py --epochs 20 --lr 2e-4
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

from source_prior import SourcePriorDenoiser

IMG_H, IMG_W = 28, 28
CKPT_DIR = "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--P_mean", type=float, default=-1.2)
    parser.add_argument("--P_std", type=float, default=1.2)
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(CKPT_DIR, "denoiser.pt"))
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Fashion-MNIST train set ...")
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=True, download=True,
        transform=T.ToTensor())
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    print(f"  images: {tuple(images.shape)}  dtype={images.dtype}")

    loader = DataLoader(TensorDataset(images), batch_size=args.batch,
                        shuffle=True, drop_last=True)

    print("Building model ...")
    model = SourcePriorDenoiser(
        img_h=IMG_H, img_w=IMG_W, bits_per_pixel=8,
        model_channels=64, channel_mult=(1, 2, 2),
        num_blocks=2, attn_resolutions=(7,),
        sigma_data=args.sigma_data,
    ).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  |  Device: {device}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print()

    opt = torch.optim.Adam(model.net.parameters(), lr=args.lr)
    sd = args.sigma_data

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for (x,) in loader:
            x = x.to(device)
            B = x.shape[0]

            ln_sigma = torch.randn(B, device=device) * args.P_std + args.P_mean
            sigma = ln_sigma.exp()

            noise = torch.randn_like(x)
            noisy = x + sigma.reshape(-1, 1, 1, 1) * noise

            denoised = model.net(noisy, sigma)

            weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
            loss = (weight.reshape(-1, 1, 1, 1) * (denoised - x) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        elapsed = time.time() - t0
        avg = np.mean(losses)
        print(f"Epoch {epoch:>2}/{args.epochs}  "
              f"loss={avg:.4f}  ({elapsed:.0f}s, {len(losses)} steps)")

    torch.save(model.state_dict(), args.ckpt)
    print(f"\nSaved → {args.ckpt}")


if __name__ == "__main__":
    main()
