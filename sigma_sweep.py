"""
Sigma sweep: baseline BP vs BP+Denoiser for multiple σ values.

Fixed alpha/beta, sweep sigma over EBNO_LIST.

Usage:
  CUDA_VISIBLE_DEVICES=0 python sigma_sweep.py
  CUDA_VISIBLE_DEVICES=0 python sigma_sweep.py --alpha 0.1 --beta 0.0
  CUDA_VISIBLE_DEVICES=0 python sigma_sweep.py --sigmas 0.1 0.2 0.3 0.5 1.0
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
BP_SCHEDULE = [10, 10, 10]
BATCH = 200
ROUNDS = 5
SEED = 42
CKPT = "checkpoints/denoiser.pt"

EBNO_LIST = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
DEFAULT_SIGMAS = [0.1, 0.2, 0.3, 0.5, 1.0]


def build_test_bitbank():
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def run_sweep(ebno_list, dec_base, dec_dn, ldpc_enc, crc_enc, crc_dec,
              mapper, demapper, awgn, bit_bank):
    nack_base = []
    nack_dn = []
    ber_base = []
    ber_dn = []

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
        nack_base.append((total - ab) / total)
        nack_dn.append((total - ad) / total)
        ber_base.append(eb / (total * K_PAYLOAD))
        ber_dn.append(ed / (total * K_PAYLOAD))

    return nack_base, nack_dn, ber_base, ber_dn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--ebno", type=float, nargs="+", default=EBNO_LIST)
    args = p.parse_args()

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()

    bit_bank = build_test_bitbank()
    print(f"Test bitbank: {bit_bank.shape}  (Fashion-MNIST test, {IMG_H}x{IMG_W}, {BPP}-bit)")
    print(f"α={args.alpha}  β={args.beta}  schedule={BP_SCHEDULE}")
    print(f"σ sweep: {args.sigmas}\n")

    dec_base = LDPC5GDecoder(
        ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(BP_SCHEDULE), llr_max=30.0)

    # ---- baseline (run once) ----
    print("Running baseline BP ...")
    baseline_nack = []
    baseline_ber = []
    for ebno in args.ebno:
        no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
        ab = eb = 0
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
        total = BATCH * ROUNDS
        baseline_nack.append((total - ab) / total)
        baseline_ber.append(eb / (total * K_PAYLOAD))
        print(f"  Eb/N0={ebno:+.1f}  BLER={baseline_nack[-1]:.4f}  BER={baseline_ber[-1]:.2e}")

    # ---- sigma sweep ----
    all_nack_dn = {}
    all_ber_dn = {}

    for sigma in args.sigmas:
        print(f"\nσ = {sigma} ...")
        dec_dn = LDPC5GDecoder_soft(
            ldpc_enc, bp_schedule=BP_SCHEDULE,
            alpha=args.alpha, beta=args.beta, k_payload=K_PAYLOAD,
            cn_update="boxplus-phi", vn_update="sum",
            cn_schedule="flooding", hard_out=False, return_infobits=True,
            num_iter=sum(BP_SCHEDULE), llr_max=30.0)

        if os.path.isfile(args.ckpt):
            dec_dn.denoiser.load_weights_pt(args.ckpt)
        else:
            print(f"  [WARN] {args.ckpt} not found — using random weights")
        dec_dn.denoiser.sigma = sigma

        nack_dn = []
        ber_dn = []
        for ebno in args.ebno:
            no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
            ad = ed = 0
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
                hat_d = dec_dn(llr_ch)
                _, cv_d = crc_dec(hat_d)
                ad += int(tf.reduce_sum(tf.cast(cv_d, tf.int32)).numpy())
                ed += int(tf.reduce_sum(tf.cast(
                    tf.not_equal(u, tf.cast(hat_d[:, :K_PAYLOAD] > 0, tf.int32)),
                    tf.int32)).numpy())
            total = BATCH * ROUNDS
            nack_dn.append((total - ad) / total)
            ber_dn.append(ed / (total * K_PAYLOAD))
            print(f"  Eb/N0={ebno:+.1f}  BLER={nack_dn[-1]:.4f}  BER={ber_dn[-1]:.2e}")

        all_nack_dn[sigma] = nack_dn
        all_ber_dn[sigma] = ber_dn

    # ---- plot ----
    os.makedirs("results", exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.sigmas)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(args.ebno, baseline_nack, "o--", color="red",
                 linewidth=2, markersize=6, label=f"Baseline BP ({sum(BP_SCHEDULE)} iter)")
    ax2.semilogy(
        [e for e, v in zip(args.ebno, baseline_ber) if v > 0],
        [v for v in baseline_ber if v > 0],
        "o--", color="red", linewidth=2, markersize=6, label="Baseline BP")

    for (sigma, nack), (_, ber), color in zip(
            all_nack_dn.items(), all_ber_dn.items(), colors):
        ax1.semilogy(args.ebno, nack, "s-", color=color,
                     linewidth=2, markersize=6, label=f"σ={sigma}")
        ber_pos = [(e, v) for e, v in zip(args.ebno, ber) if v > 0]
        if ber_pos:
            eb_p, er_p = zip(*ber_pos)
            ax2.semilogy(eb_p, er_p, "s-", color=color,
                         linewidth=2, markersize=6, label=f"σ={sigma}")

    for ax in (ax1, ax2):
        ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("BLER", fontsize=12)
    ax1.set_title("Block Error Rate", fontsize=13)
    ax1.set_ylim(bottom=5e-4)
    ax2.set_ylabel("BER", fontsize=12)
    ax2.set_title("Bit Error Rate", fontsize=13)

    fig.suptitle(
        f"Sigma sweep — α={args.alpha}, β={args.beta}, schedule={BP_SCHEDULE}\n"
        f"Fashion-MNIST, K={K_PAYLOAD}, N={N_CODEWORD}, BPSK, "
        f"new_input = ch + β·bp_ext + α·src_ext",
        fontsize=10, y=1.02)
    fig.tight_layout()

    out = f"results/sigma_sweep_a{args.alpha}_b{args.beta}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
