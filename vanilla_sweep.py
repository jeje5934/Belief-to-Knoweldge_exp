"""
vanilla_sweep.py -- BP + source-denoiser, fixed sigma=0.3.

No adaptive sigma. No syndrome lookup. No sigma calibration.

  new_input = channel + beta * bp_ext + alpha_t * src_ext

  bp_ext  = BP_post - payload_intr      (BP turbo extrinsic)
  src_ext = src_post - BP_post          (source extrinsic, SoftDenoiser output)

CLI reference
-------------
Smoke test:

  CUDA_VISIBLE_DEVICES=0 python vanilla_sweep.py \
      --ebno 0.7 \
      --betas -0.10 0.00 0.10 \
      --alpha-schedules "0.10,0.10" "0.15,0.10" \
      --sigma 0.3 --batch 32 --rounds 1 --tag smoke

Conservative main sweep:

  CUDA_VISIBLE_DEVICES=0 python vanilla_sweep.py \
      --ebno 0.5 0.6 0.7 0.8 \
      --betas -0.20 -0.15 -0.10 -0.05 0.00 0.05 0.10 \
      --alpha-schedules "0.05,0.05" "0.10,0.10" "0.15,0.15" "0.15,0.10" "0.20,0.10" "0.10,0.05" \
      --sigma 0.3 --batch 64 --rounds 2 --tag main_sweep

--alpha is used only when --alpha-schedules is omitted, as a constant schedule.
"""
import argparse
import csv
import os
from datetime import datetime, timezone

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

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

import torch
import torchvision

# ---------- Fixed constants --------------------------------------------------
IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP   # 6272 bits

CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1                         # BPSK
DEFAULT_BP_SCHEDULE = [10, 10, 10]
FIXED_SIGMA = 0.3
DEFAULT_CKPT = "checkpoints/denoiser.pt"
SEED = 42

CSV_FIELDS = [
    "timestamp", "tag", "mode",
    "ebno_db", "sigma", "beta", "alpha_schedule",
    "bler", "ber",
    "n_blocks", "n_acks", "n_nacks", "n_bit_errors",
    "bp_schedule",
]


# ---------- Helpers ----------------------------------------------------------

def build_test_bitbank():
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def parse_alpha_schedule(s):
    parts = s.split(",")
    return [float(x.strip()) for x in parts if x.strip()]


def fmt_schedule(sched):
    return "[" + ",".join("{:.2f}".format(a) for a in sched) + "]"


def fmt_bp_schedule(bp):
    return "[" + ",".join(str(x) for x in bp) + "]"


def run_one(dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, ebno, batch, rounds):
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    tot_ack = tot_err = 0
    n_blocks = batch * rounds
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
        tot_ack += int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_err += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
    n_nacks = n_blocks - tot_ack
    bler = n_nacks / n_blocks
    ber = tot_err / (n_blocks * K_PAYLOAD)
    return tot_ack, n_nacks, tot_err, n_blocks, bler, ber


def append_csv(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def make_row(ts, tag, mode, ebno, sigma, beta, alpha_sched_str,
             bler, ber, n_acks, n_nacks, n_bit_errors, n_blocks, bp_sched_str):
    return {
        "timestamp": ts,
        "tag": tag,
        "mode": mode,
        "ebno_db": "{:.2f}".format(ebno),
        "sigma": sigma,
        "beta": beta,
        "alpha_schedule": alpha_sched_str,
        "bler": "{:.6f}".format(bler),
        "ber": "{:.6e}".format(ber),
        "n_blocks": n_blocks,
        "n_acks": n_acks,
        "n_nacks": n_nacks,
        "n_bit_errors": n_bit_errors,
        "bp_schedule": bp_sched_str,
    }


# ---------- CLI --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Vanilla BP+denoiser sweep (fixed sigma=0.3, no adaptive sigma).")

    p.add_argument("--ebno", type=float, nargs="+",
                   default=[0.5, 0.6, 0.7, 0.8], metavar="DB",
                   help="Eb/N0 value(s) in dB")
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.0], metavar="BETA",
                   help="Beta values to sweep (negative values accepted)")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="Scalar alpha used as constant schedule when "
                        "--alpha-schedules is not given")
    p.add_argument("--alpha-schedules", type=str, nargs="+", default=None,
                   metavar="SCHED",
                   help='Alpha schedules as comma-separated floats, e.g. '
                        '"0.10,0.10" "0.15,0.10" "0.20,0.10"')
    p.add_argument("--sigma", type=float, default=FIXED_SIGMA,
                   help="Denoiser sigma (vanilla branch: keep at 0.3)")
    p.add_argument("--bp-schedule", type=int, nargs="+",
                   default=DEFAULT_BP_SCHEDULE, metavar="ITER",
                   help="BP iterations per chunk (default: 10 10 10)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--csv", default="results/vanilla_sweep.csv",
                   help="Output CSV path (rows appended)")
    p.add_argument("--tag", default="",
                   help="Sweep tag written to every CSV row")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip pure-BP baseline run")
    return p.parse_args()


# ---------- Main -------------------------------------------------------------

def main():
    args = parse_args()

    if args.sigma != FIXED_SIGMA:
        print("[WARN] --sigma={} differs from vanilla default {}. "
              "This branch uses fixed sigma only.".format(args.sigma, FIXED_SIGMA))

    # Resolve alpha schedules.
    n_denoiser_calls = max(len(args.bp_schedule) - 1, 0)

    if args.alpha_schedules is not None:
        alpha_schedules = [parse_alpha_schedule(s) for s in args.alpha_schedules]
    else:
        const_sched = [args.alpha] * max(n_denoiser_calls, 1)
        alpha_schedules = [const_sched]

    bp_sched_str = fmt_bp_schedule(args.bp_schedule)

    SEP = "=" * 72
    print(SEP)
    print("vanilla_sweep  --  fixed sigma={}, no adaptive sigma".format(args.sigma))
    print("  BP schedule    : {}  ({} denoiser call(s))".format(
        bp_sched_str, n_denoiser_calls))
    print("  Eb/N0          : {}".format(args.ebno))
    print("  Beta values    : {}".format(args.betas))
    print("  Alpha schedules:")
    for s in alpha_schedules:
        print("    {}".format(fmt_schedule(s)))
    print("  batch={}  rounds={}  blocks/config/ebno={}".format(
        args.batch, args.rounds, args.batch * args.rounds))
    print("  tag={!r}  csv={}".format(args.tag, args.csv))
    print(SEP)

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()
    bit_bank = build_test_bitbank()

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load checkpoint once into CPU memory; reuse across all decoder instances
    # to avoid repeated file I/O and PyTorch serialisation state issues.
    ckpt_state = None
    if os.path.isfile(args.ckpt):
        ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        print("Checkpoint loaded: {}".format(args.ckpt))
    else:
        print("[WARN] {} not found -- all denoiser runs use random weights".format(
            args.ckpt))

    # ---- Baseline BP (no denoiser) -----------------------------------------
    if not args.no_baseline:
        print("\n[Baseline BP -- no denoiser]")
        dec_base = LDPC5GDecoder(
            ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
            cn_schedule="flooding", hard_out=False, return_infobits=True,
            num_iter=sum(args.bp_schedule), llr_max=30.0)

        for ebno in args.ebno:
            n_acks, n_nacks, n_bit_err, n_blk, bler, ber = run_one(
                dec_base, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, ebno, args.batch, args.rounds)
            print("  Eb/N0={:+.2f}  BLER={:.4f}  BER={:.2e}  ACK={}/{}".format(
                ebno, bler, ber, n_acks, n_blk))
            append_csv(args.csv, make_row(
                ts, args.tag, "baseline",
                ebno, "N/A", "N/A", "N/A",
                bler, ber, n_acks, n_nacks, n_bit_err, n_blk, bp_sched_str))

        del dec_base

    # ---- Beta x alpha-schedule sweep ---------------------------------------
    total_configs = len(args.betas) * len(alpha_schedules)
    cfg_idx = 0

    for beta in args.betas:
        for alpha_sched in alpha_schedules:
            cfg_idx += 1
            sched_tag = fmt_schedule(alpha_sched)
            print("\n[{}/{}]  beta={:+.3f}  alpha_schedule={}  sigma={}".format(
                cfg_idx, total_configs, beta, sched_tag, args.sigma))

            dec_dn = LDPC5GDecoder_soft(
                ldpc_enc,
                bp_schedule=args.bp_schedule,
                alpha=alpha_sched[0],
                beta=beta,
                alpha_schedule=alpha_sched,
                k_payload=K_PAYLOAD,
                cn_update="boxplus-phi", vn_update="sum",
                cn_schedule="flooding", hard_out=False, return_infobits=True,
                num_iter=sum(args.bp_schedule), llr_max=30.0)

            if ckpt_state is not None:
                dec_dn.denoiser.prior_model.load_state_dict(ckpt_state)
                dec_dn.denoiser.prior_model.eval()
            else:
                print("  [WARN] no checkpoint -- using random weights")

            # Fixed sigma: vanilla branch, do not change.
            dec_dn.denoiser.sigma = args.sigma

            for ebno in args.ebno:
                n_acks, n_nacks, n_bit_err, n_blk, bler, ber = run_one(
                    dec_dn, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                    bit_bank, ebno, args.batch, args.rounds)
                print("  Eb/N0={:+.2f}  BLER={:.4f}  BER={:.2e}  ACK={}/{}".format(
                    ebno, bler, ber, n_acks, n_blk))
                append_csv(args.csv, make_row(
                    ts, args.tag, "denoiser",
                    ebno, args.sigma, beta, sched_tag,
                    bler, ber, n_acks, n_nacks, n_bit_err, n_blk, bp_sched_str))

            del dec_dn

    print("\n[Done]  {} config(s) x {} Eb/N0 point(s).".format(
        cfg_idx, len(args.ebno)))
    print("Results appended to: {}".format(args.csv))


if __name__ == "__main__":
    main()
