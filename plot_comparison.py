"""
Eb/N0 sweep: baseline BP vs onlyextrinsic BP+Denoiser (alpha/beta).

[onlyextrinsic variant — independent alpha/beta]
  new_input = channel + beta * bp_ext + alpha * src_ext

Uses Fashion-MNIST **test set** (10,000 images, 28×28 grayscale) for
evaluation. The denoiser is trained on the disjoint train set.

Saves plot to results/comparison.png.

Usage:
  CUDA_VISIBLE_DEVICES=0 python plot_comparison.py
  CUDA_VISIBLE_DEVICES=0 python plot_comparison.py --alpha 0.1 --beta 0.1
  CUDA_VISIBLE_DEVICES=0 python plot_comparison.py --alpha 0.1 --beta 0.0 --sigma 0.3
"""
import argparse
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

import torchvision

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP  # 6272

CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1

BEST_ALPHA = 0.1
BEST_BETA = 0.1
BEST_SIGMA = 0.3
BP_SCHEDULE = [10, 10, 10]
BATCH = 200
ROUNDS = 5
SEED = 42
CKPT = "checkpoints/denoiser.pt"

EBNO_LIST = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]


def build_test_bitbank():
    """Fashion-MNIST test set (10,000 images), grayscale 28x28 → 6272 bits each."""
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def run_sweep(ebno_list, dec_base, dec_dn, ldpc_enc, crc_enc, crc_dec,
              mapper, demapper, awgn, bit_bank):
    results = {"ebno": [], "nack_base": [], "nack_dn": [],
               "err_base": [], "err_dn": []}

    for ebno in ebno_list:
        no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
        ab = eb = ad = ed = 0

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

            hat_b = dec_base(llr_ch)
            _, cv_b = crc_dec(hat_b)
            ab += int(tf.reduce_sum(tf.cast(cv_b, tf.int32)).numpy())
            eb += int(tf.reduce_sum(tf.cast(
                tf.not_equal(u, tf.cast(hat_b[:, :K_PAYLOAD] > 0, tf.int32)),
                tf.int32)).numpy())

            hat_d = dec_dn(llr_ch)
            _, cv_d = crc_dec(hat_d)
            ad += int(tf.reduce_sum(tf.cast(cv_d, tf.int32)).numpy())
            ed += int(tf.reduce_sum(tf.cast(
                tf.not_equal(u, tf.cast(hat_d[:, :K_PAYLOAD] > 0, tf.int32)),
                tf.int32)).numpy())

        total = BATCH * ROUNDS
        results["ebno"].append(ebno)
        results["nack_base"].append((total - ab) / total)
        results["nack_dn"].append((total - ad) / total)
        results["err_base"].append(eb / (total * K_PAYLOAD))
        results["err_dn"].append(ed / (total * K_PAYLOAD))

        print(f"Eb/N0={ebno:+.1f} dB  |  "
              f"Baseline NACK={total-ab:>4}/{total}  BER={eb/(total*K_PAYLOAD):.2e}  |  "
              f"Denoiser NACK={total-ad:>4}/{total}  BER={ed/(total*K_PAYLOAD):.2e}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=BEST_ALPHA)
    p.add_argument("--beta", type=float, default=BEST_BETA)
    p.add_argument("--sigma", type=float, default=BEST_SIGMA)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--ebno", type=float, nargs="+", default=EBNO_LIST)
    args = p.parse_args()

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    dec_base = LDPC5GDecoder(
        ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(BP_SCHEDULE), llr_max=30.0)

    dec_dn = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=BP_SCHEDULE,
        alpha=args.alpha, beta=args.beta, k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(BP_SCHEDULE), llr_max=30.0)

    if os.path.isfile(args.ckpt):
        dec_dn.denoiser.load_weights_pt(args.ckpt)
        print(f"[INFO] Loaded {args.ckpt}")
    else:
        print(f"[WARN] {args.ckpt} not found")
    dec_dn.denoiser.sigma = args.sigma

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()

    bit_bank = build_test_bitbank()
    print(f"Test bitbank: {bit_bank.shape} (Fashion-MNIST test set, "
          f"grayscale {IMG_H}x{IMG_W}, {BPP}-bit → {K_PAYLOAD} bits/img)")
    print(f"α={args.alpha}  β={args.beta}  σ={args.sigma}  "
          f"schedule={BP_SCHEDULE}\n")

    res = run_sweep(args.ebno, dec_base, dec_dn, ldpc_enc, crc_enc,
                    crc_dec, mapper, demapper, awgn, bit_bank)

    # ---- plot ----
    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    dn_label = (f"BP + Denoiser (α={args.alpha}, β={args.beta}, "
                f"σ={args.sigma})")

    # NACK rate (= BLER)
    ax1.semilogy(res["ebno"], res["nack_base"], "o-", color="#d62728",
                 linewidth=2, markersize=7,
                 label=f"Baseline BP ({sum(BP_SCHEDULE)} iter)")
    ax1.semilogy(res["ebno"], res["nack_dn"], "s-", color="#1f77b4",
                 linewidth=2, markersize=7, label=dn_label)
    ax1.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax1.set_ylabel("NACK Rate (BLER)", fontsize=12)
    ax1.set_title("Block Error Rate", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(bottom=5e-4)

    # BER
    mask_b = [v > 0 for v in res["err_base"]]
    mask_d = [v > 0 for v in res["err_dn"]]
    eb_b = [e for e, m in zip(res["ebno"], mask_b) if m]
    er_b = [e for e, m in zip(res["err_base"], mask_b) if m]
    eb_d = [e for e, m in zip(res["ebno"], mask_d) if m]
    er_d = [e for e, m in zip(res["err_dn"], mask_d) if m]

    if er_b:
        ax2.semilogy(eb_b, er_b, "o-", color="#d62728",
                     linewidth=2, markersize=7, label="Baseline BP")
    if er_d:
        ax2.semilogy(eb_d, er_d, "s-", color="#1f77b4",
                     linewidth=2, markersize=7,
                     label=f"BP + Denoiser (α={args.alpha}, β={args.beta})")
    ax2.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax2.set_ylabel("Bit Error Rate", fontsize=12)
    ax2.set_title("Bit Error Rate", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Fashion-MNIST over AWGN — Baseline BP vs onlyextrinsic BP+Denoiser\n"
        f"new_input = ch + β·bp_ext + α·src_ext  |  "
        f"K={K_PAYLOAD}, N={N_CODEWORD}, BPSK, schedule={BP_SCHEDULE}, test set",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out = "results/comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
