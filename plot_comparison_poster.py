"""
plot_comparison_poster.py — Eb/N0 sweep: multiple Baseline BP vs Proposed [5x6] (Poster Version).
Only plots clean graphs suitable for poster presentation.
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
DEFAULT_ALPHA_SCHED   = "0.10,0.08,0.06,0.04,0.02"
DEFAULT_BETA          = 0.1
DEFAULT_SIGMA         = 0.3
DEFAULT_EBNO          = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_BATCH         = 64
DEFAULT_ROUNDS        = 5
DEFAULT_BASELINES     = [30]   # pure-BP iteration counts to plot

# Premium color palette
BASELINE_COLOURS = ["#E15759", "#F28E2B", "#499894", "#86BCB6"]
BASELINE_MARKERS = ["o", "^", "D", "v"]
PROPOSED_COLOR = "#4E79A7" # Sleek Classic Blue

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

def eval_decoder(dec, ebno_list, ldpc_enc, crc_enc, crc_dec,
                 mapper, demapper, awgn, bit_bank, batch, rounds,
                 label=""):
    """Run one decoder over all Eb/N0 points. Returns (nack_list, ber_list)."""
    nack_list, ber_list = [], []
    total = batch * rounds

    for ebno in ebno_list:
        no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
        n_ack = n_err = 0

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

            hat = dec(llr_ch)
            _, cv = crc_dec(hat)
            n_ack += int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
            n_err += int(tf.reduce_sum(tf.cast(
                tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
                tf.int32)).numpy())

        nack = (total - n_ack) / total
        ber  = n_err / (total * K_PAYLOAD)
        nack_list.append(nack)
        ber_list.append(ber)
        tag = f"[{label}]" if label else ""
        print(f"  Eb/N0={ebno:+.2f}  {tag}  "
              f"BLER={nack:.4f}  BER={ber:.2e}  (ACK {n_ack}/{total})")

    return nack_list, ber_list

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Plot multiple Baseline BP vs Proposed [5x6] BLER/BER curves (Poster version).")

    p.add_argument("--bp-schedule", type=int, nargs="+",
                   default=DEFAULT_BP_SCHEDULE, metavar="ITER")
    p.add_argument("--alpha-schedule", type=str,
                   default=DEFAULT_ALPHA_SCHED, metavar="SCHED")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta",  type=float, default=DEFAULT_BETA)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--ckpt",  default=CKPT)
    p.add_argument("--baselines", type=int, nargs="+",
                   default=DEFAULT_BASELINES, metavar="ITER")
    p.add_argument("--ebno", type=float, nargs="+",
                   default=DEFAULT_EBNO, metavar="DB")
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--out", default="results/comparison_5x6_beta01.png",
                   help="Output PNG path")
    return p.parse_args()

def main():
    args = parse_args()

    if args.alpha is not None:
        n_calls = max(len(args.bp_schedule) - 1, 1)
        alpha_sched = [args.alpha] * n_calls
    else:
        alpha_sched = parse_alpha_schedule(args.alpha_schedule)

    bp_total = sum(args.bp_schedule)

    print("=" * 68)
    print("plot_comparison_poster (Poster Version)")
    print(f"  Baselines      : {['BP-' + str(n) for n in args.baselines]}")
    print(f"  BP schedule    : {args.bp_schedule}  ({bp_total} total iter)")
    print(f"  alpha schedule : {fmt_sched(alpha_sched)}")
    print(f"  beta           : {args.beta}")
    print(f"  sigma          : {args.sigma}")
    print(f"  Eb/N0          : {args.ebno}")
    print(f"  batch={args.batch}  rounds={args.rounds}")
    print("=" * 68)

    # Sionna setup
    crc_enc  = CRCEncoder(CRC_DEGREE)
    crc_dec  = CRCDecoder(crc_enc)
    k_ldpc   = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper   = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn     = AWGN()
    bit_bank = build_test_bitbank()

    # Baseline decoders
    baseline_decoders = {}
    for n_iter in sorted(set(args.baselines)):
        baseline_decoders[n_iter] = LDPC5GDecoder(
            ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
            cn_schedule="flooding", hard_out=False, return_infobits=True,
            num_iter=n_iter, llr_max=30.0)

    # Proposed
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

    if os.path.isfile(args.ckpt):
        ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        dec_dn.denoiser.prior_model.load_state_dict(ckpt_state)
        dec_dn.denoiser.prior_model.eval()
        print(f"Checkpoint loaded: {args.ckpt}")
    else:
        print(f"[WARN] {args.ckpt} not found — using random weights")
    dec_dn.denoiser.sigma = args.sigma

    # Run sweeps
    baseline_results = {}
    for n_iter in sorted(set(args.baselines)):
        print(f"\n--- Baseline BP-{n_iter} ---")
        nack, ber = eval_decoder(
            baseline_decoders[n_iter], args.ebno,
            ldpc_enc, crc_enc, crc_dec,
            mapper, demapper, awgn, bit_bank,
            args.batch, args.rounds, label=f"BP-{n_iter}")
        baseline_results[n_iter] = (nack, ber)

    print(f"\n--- Proposed {args.bp_schedule} ---")
    nack_dn, ber_dn = eval_decoder(
        dec_dn, args.ebno, ldpc_enc, crc_enc, crc_dec,
        mapper, demapper, awgn, bit_bank,
        args.batch, args.rounds, label="Proposed")

    # Plot Setup for Poster
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'grid.alpha': 0.4
    })

    def pos(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
        return (list(a) for a in zip(*pairs)) if pairs else ([], [])

    # Determine individual output filenames
    out_dir = os.path.dirname(os.path.abspath(args.out))
    out_base = os.path.basename(args.out)
    name, ext = os.path.splitext(out_base)
    bler_out = os.path.join(out_dir, f"{name}_bler{ext}")
    ber_out = os.path.join(out_dir, f"{name}_ber{ext}")

    # --- 1. Plot BLER ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    for i, n_iter in enumerate(sorted(set(args.baselines))):
        nack, _ = baseline_results[n_iter]
        colour  = BASELINE_COLOURS[i % len(BASELINE_COLOURS)]
        marker  = BASELINE_MARKERS[i % len(BASELINE_MARKERS)]
        ax1.semilogy(args.ebno, nack, marker + "--", color=colour,
                     linewidth=2.5, markersize=9, alpha=0.85)

    # Proposed
    ax1.semilogy(args.ebno, nack_dn, "s-", color=PROPOSED_COLOR,
                 linewidth=3.5, markersize=10)

    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Block Error Rate (BLER)")
    ax1.grid(True, which="both")
    ax1.set_ylim(bottom=1e-3)

    fig1.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig1.savefig(bler_out, dpi=300, bbox_inches="tight")
    print(f"\nBLER plot saved → {bler_out}")
    plt.close(fig1)

    # --- 2. Plot BER ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for i, n_iter in enumerate(sorted(set(args.baselines))):
        _, ber = baseline_results[n_iter]
        colour  = BASELINE_COLOURS[i % len(BASELINE_COLOURS)]
        marker  = BASELINE_MARKERS[i % len(BASELINE_MARKERS)]
        eb, er = pos(args.ebno, ber)
        if er:
            ax2.semilogy(list(eb), list(er), marker + "--", color=colour,
                         linewidth=2.5, markersize=9, alpha=0.85)

    # Proposed
    eb_d, er_d = pos(args.ebno, ber_dn)
    if er_d:
        ax2.semilogy(list(eb_d), list(er_d), "s-", color=PROPOSED_COLOR,
                     linewidth=3.5, markersize=10)

    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Bit Error Rate (BER)")
    ax2.grid(True, which="both")
    ax2.set_ylim(bottom=1e-6)

    fig2.tight_layout()
    fig2.savefig(ber_out, dpi=300, bbox_inches="tight")
    print(f"\nBER plot saved → {ber_out}")
    plt.close(fig2)

if __name__ == "__main__":
    main()
