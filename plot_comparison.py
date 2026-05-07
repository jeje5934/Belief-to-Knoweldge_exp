"""
plot_comparison.py — Eb/N0 sweep: Baseline BP-30 vs Proposed [5x6].

Main paper figure:
  Baseline : pure BP, 30 total BP iterations ([5x6] same budget)
  Proposed : [5,5,5,5,5,5], sigma=0.3, alpha=[0.10×5], beta=0.1

  new_input = channel + beta * bp_ext + alpha_t * src_ext
  bp_ext  = BP_post - payload_intr
  src_ext = src_post - BP_post          (returned by SoftDenoiser)

Usage:
  # Default (paper main figure)
  python plot_comparison.py

  # Custom schedule / knobs
  python plot_comparison.py \\
      --bp-schedule 5 5 5 5 5 5 \\
      --alpha-schedule "0.10,0.10,0.10,0.10,0.10" \\
      --beta 0.1 --sigma 0.3 \\
      --ebno 0.4 0.5 0.6 0.7 0.8 0.9 1.0

  # Quick smoke (small batch)
  python plot_comparison.py --batch 32 --rounds 1
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
import torch
import torchvision

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

# ── Constants ────────────────────────────────────────────────────────────────
IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP   # 6272 bits

CRC_DEGREE  = "CRC24A"
N_CODEWORD  = 12600
NUM_BPS     = 1                   # BPSK
SEED        = 42
CKPT        = "checkpoints/denoiser.pt"

# ── Paper-default hyper-parameters ───────────────────────────────────────────
DEFAULT_BP_SCHEDULE   = [5, 5, 5, 5, 5, 5]        # 30 total BP iter
DEFAULT_ALPHA_SCHED   = "0.10,0.10,0.10,0.10,0.10"
DEFAULT_BETA          = 0.1
DEFAULT_SIGMA         = 0.3
DEFAULT_EBNO          = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_BATCH         = 200
DEFAULT_ROUNDS        = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_alpha_schedule(s: str) -> list:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def fmt_sched(sched: list) -> str:
    return "[" + ",".join(f"{a:.2f}" for a in sched) + "]"


def build_test_bitbank():
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def run_sweep(ebno_list, dec_base, dec_dn, ldpc_enc, crc_enc, crc_dec,
              mapper, demapper, awgn, bit_bank, batch, rounds):
    nack_base, nack_dn, ber_base, ber_dn = [], [], [], []
    total = batch * rounds

    for ebno in ebno_list:
        no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
        ab = eb = ad = ed = 0

        for r in range(rounds):
            tf.random.set_seed(SEED + r + int(ebno * 1000))
            n_imgs = tf.shape(bit_bank)[0]
            idx = tf.random.uniform([batch], 0, n_imgs, dtype=tf.int32)
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

        nack_base.append((total - ab) / total)
        nack_dn.append((total - ad) / total)
        ber_base.append(eb / (total * K_PAYLOAD))
        ber_dn.append(ed / (total * K_PAYLOAD))

        print(f"  Eb/N0={ebno:+.2f}  "
              f"Baseline BLER={nack_base[-1]:.4f} BER={ber_base[-1]:.2e}  |  "
              f"Proposed BLER={nack_dn[-1]:.4f} BER={ber_dn[-1]:.2e}  "
              f"(ACK {ad}/{total})")

    return nack_base, nack_dn, ber_base, ber_dn


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Plot Baseline BP-30 vs Proposed [5x6] BLER/BER curves.")

    p.add_argument("--bp-schedule", type=int, nargs="+",
                   default=DEFAULT_BP_SCHEDULE, metavar="ITER",
                   help="BP iterations per chunk (default: 5 5 5 5 5 5)")
    p.add_argument("--alpha-schedule", type=str,
                   default=DEFAULT_ALPHA_SCHED, metavar="SCHED",
                   help='Per-call alpha as comma-separated string '
                        '(default: "0.10,0.10,0.10,0.10,0.10")')
    p.add_argument("--alpha", type=float, default=None,
                   help="Scalar alpha (overrides --alpha-schedule with constant)")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--ebno", type=float, nargs="+",
                   default=DEFAULT_EBNO, metavar="DB")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--out", default="results/comparison.png",
                   help="Output PNG path")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve alpha schedule
    if args.alpha is not None:
        n_denoiser = max(len(args.bp_schedule) - 1, 0)
        alpha_sched = [args.alpha] * max(n_denoiser, 1)
    else:
        alpha_sched = parse_alpha_schedule(args.alpha_schedule)

    bp_total = sum(args.bp_schedule)
    n_denoiser_calls = max(len(args.bp_schedule) - 1, 0)

    print("=" * 68)
    print("plot_comparison — Baseline BP-30 vs Proposed [5x6]")
    print(f"  BP schedule    : {args.bp_schedule}  ({bp_total} total iter)")
    print(f"  Denoiser calls : {n_denoiser_calls}")
    print(f"  alpha schedule : {fmt_sched(alpha_sched)}")
    print(f"  beta           : {args.beta}")
    print(f"  sigma          : {args.sigma}  (fixed)")
    print(f"  Eb/N0          : {args.ebno}")
    print(f"  batch={args.batch}  rounds={args.rounds}  "
          f"blocks/ebno={args.batch * args.rounds}")
    print("=" * 68)

    # ── Sionna setup ─────────────────────────────────────────────────────────
    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc  = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper   = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    awgn     = AWGN()
    bit_bank = build_test_bitbank()

    # ── Baseline: pure BP, same total iter ───────────────────────────────────
    dec_base = LDPC5GDecoder(
        ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=bp_total, llr_max=30.0)

    # ── Proposed: [5x6] + source-prior extrinsic ─────────────────────────────
    dec_dn = LDPC5GDecoder_soft(
        ldpc_enc,
        bp_schedule=args.bp_schedule,
        alpha=alpha_sched[0],
        beta=args.beta,
        alpha_schedule=alpha_sched,
        k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=bp_total, llr_max=30.0)

    # Load checkpoint once into memory, apply via load_state_dict
    if os.path.isfile(args.ckpt):
        ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        dec_dn.denoiser.prior_model.load_state_dict(ckpt_state)
        dec_dn.denoiser.prior_model.eval()
        print(f"Checkpoint loaded: {args.ckpt}")
    else:
        print(f"[WARN] {args.ckpt} not found — using random weights")

    dec_dn.denoiser.sigma = args.sigma

    # ── Run sweep ─────────────────────────────────────────────────────────────
    print()
    nack_base, nack_dn, ber_base, ber_dn = run_sweep(
        args.ebno, dec_base, dec_dn, ldpc_enc, crc_enc, crc_dec,
        mapper, demapper, awgn, bit_bank, args.batch, args.rounds)

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    base_label = f"Baseline BP ({bp_total} iter)"
    prop_label = (f"Proposed: {args.bp_schedule}  "
                  f"α={fmt_sched(alpha_sched)}  β={args.beta}  σ={args.sigma}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # BLER
    ax1.semilogy(args.ebno, nack_base, "o--", color="#d62728",
                 linewidth=2, markersize=7, label=base_label)
    ax1.semilogy(args.ebno, nack_dn, "s-", color="#1f77b4",
                 linewidth=2.5, markersize=7, label=prop_label)
    ax1.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax1.set_ylabel("BLER (NACK rate)", fontsize=12)
    ax1.set_title("Block Error Rate", fontsize=13)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(bottom=1e-3)

    # BER
    def pos(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
        return zip(*pairs) if pairs else ([], [])

    eb_b, er_b = pos(args.ebno, ber_base)
    eb_d, er_d = pos(args.ebno, ber_dn)
    if er_b:
        ax2.semilogy(list(eb_b), list(er_b), "o--", color="#d62728",
                     linewidth=2, markersize=7, label=base_label)
    if er_d:
        ax2.semilogy(list(eb_d), list(er_d), "s-", color="#1f77b4",
                     linewidth=2.5, markersize=7, label=prop_label)
    ax2.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax2.set_ylabel("BER", fontsize=12)
    ax2.set_title("Bit Error Rate", fontsize=13)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Fashion-MNIST / AWGN  —  Baseline BP-30 vs Proposed Source-Prior Extrinsic\n"
        f"new_input = ch + β·bp_ext + αₜ·src_ext  |  "
        f"K={K_PAYLOAD}, N={N_CODEWORD}, BPSK  |  "
        f"{args.batch * args.rounds} blocks/point",
        fontsize=10, y=1.02)
    fig.tight_layout()

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
