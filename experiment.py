"""
Alpha/beta sweep experiment.

  new_input = channel + beta * bp_ext + alpha * src_ext

  bp_ext  = BP_post  - payload_intr   (turbo BP extrinsic, posterior - a priori)
  src_ext = src_post - BP_post        (turbo source extrinsic, posterior - denoiser input)

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiment.py
  CUDA_VISIBLE_DEVICES=0 python experiment.py --ebno 0.8 --sigma 0.3
  CUDA_VISIBLE_DEVICES=0 python experiment.py --ebno 0.7
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

import argparse
import torchvision

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP  # 6272

CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1
SCHEDULE = [10, 10, 10]
CKPT = "checkpoints/denoiser.pt"
BATCH = 200
ROUNDS = 5
SEED = 42


def build_test_bitbank():
    """Fashion-MNIST test set (10,000 images), grayscale 28x28 → 6272 bits each."""
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def run_one(dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, ebno):
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    tot_ack = tot_err = 0
    for r in range(ROUNDS):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        n_imgs = tf.shape(bit_bank)[0]
        idx = tf.random.uniform([BATCH], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bit_bank, idx)
        u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
        c = ldpc_enc(u_crc)
        x = mapper(c)
        y = awgn(x, no)
        llr_ch = demapper(y, no)
        hat = dec(llr_ch)
        _, cv = crc_dec(hat)
        tot_ack += int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_err += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
    total = BATCH * ROUNDS
    return tot_ack, total - tot_ack, tot_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ebno", type=float, default=0.8)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--ckpt", default=CKPT)
    args = p.parse_args()

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()
    bit_bank = build_test_bitbank()

    dec_base = LDPC5GDecoder(
        ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(SCHEDULE), llr_max=30.0)

    base_ack, base_nack, base_err = run_one(
        dec_base, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
        bit_bank, args.ebno)
    total = BATCH * ROUNDS

    print(f"Eb/N0={args.ebno} dB  σ={args.sigma}  schedule={SCHEDULE}")
    print(f"Baseline BP: ACK={base_ack}/{total}  NACK={base_nack}  "
          f"BER={base_err/(total*K_PAYLOAD):.2e}\n")

    alphas = [0.0, 0.02, 0.05, 0.1, 0.2]
    betas  = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]

    results = {}

    for beta in betas:
        for alpha in alphas:
            dec = LDPC5GDecoder_soft(
                ldpc_enc, bp_schedule=SCHEDULE,
                alpha=alpha, beta=beta, k_payload=K_PAYLOAD,
                cn_update="boxplus-phi", vn_update="sum",
                cn_schedule="flooding", hard_out=False, return_infobits=True,
                num_iter=sum(SCHEDULE), llr_max=30.0)
            if os.path.isfile(args.ckpt):
                dec.denoiser.load_weights_pt(args.ckpt)
            dec.denoiser.sigma = args.sigma

            ack, nack, err = run_one(
                dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, args.ebno)
            results[(alpha, beta)] = (ack, nack, err)
            bler = nack / total
            delta = ack - base_ack
            print(f"  α={alpha:<5}  β={beta:<5}  "
                  f"ACK={ack:>4}  NACK={nack:>4}  "
                  f"BLER={bler:.3f}  ΔACK={delta:+d}")

    # ---- heatmap ----
    os.makedirs("results", exist_ok=True)

    bler_grid = np.zeros((len(betas), len(alphas)))
    for bi, beta in enumerate(betas):
        for ai, alpha in enumerate(alphas):
            _, nack, _ = results[(alpha, beta)]
            bler_grid[bi, ai] = nack / total

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(bler_grid, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=max(bler_grid.max(), base_nack / total))

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel("α  (source extrinsic weight)", fontsize=11)
    ax.set_ylabel("β  (BP extrinsic weight)", fontsize=11)

    for bi in range(len(betas)):
        for ai in range(len(alphas)):
            val = bler_grid[bi, ai]
            color = "white" if val > 0.5 else "black"
            ax.text(ai, bi, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="BLER")
    base_bler = base_nack / total
    ax.set_title(
        f"BLER heatmap — Eb/N0={args.ebno} dB, σ={args.sigma}\n"
        f"Baseline BLER={base_bler:.3f} | "
        f"new_input = ch + β·bp_ext + α·src_ext",
        fontsize=10)

    out = f"results/alpha_beta_sweep_ebno{args.ebno}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")

    best_key = min(results, key=lambda k: results[k][1])
    ba, bb = best_key
    bk_ack, bk_nack, bk_err = results[best_key]
    print(f"\nBest: α={ba}, β={bb}  →  ACK={bk_ack}  NACK={bk_nack}  "
          f"BLER={bk_nack/total:.3f}  "
          f"(baseline BLER={base_bler:.3f})")


if __name__ == "__main__":
    main()
