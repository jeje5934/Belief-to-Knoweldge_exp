"""
stats_bler_ebno08.py — Eb/N0=0.8 BLER 통계 (plot_comparison_poster.py와 동일 설정).
Baseline BP vs Proposed [5x6] 각각에 대해 round별 BLER 및 신뢰구간을 출력한다.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
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

# ── Constants (plot_comparison_poster.py와 동일) ─────────────────────────────
IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP

CRC_DEGREE  = "CRC24A"
N_CODEWORD  = 12600
NUM_BPS     = 1
SEED        = 42
CKPT        = "checkpoints/denoiser.pt"

DEFAULT_BP_SCHEDULE   = [5, 5, 5, 5, 5, 5]
DEFAULT_ALPHA_SCHED   = "0.10,0.08,0.06,0.04,0.02"
DEFAULT_BETA          = 0.1
DEFAULT_SIGMA         = 0.3
DEFAULT_EBNO          = 0.8
DEFAULT_BATCH         = 64
DEFAULT_ROUNDS        = 20
DEFAULT_BASELINES     = [30]


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


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def eval_bler_stats(dec, ebno, ldpc_enc, crc_enc, crc_dec,
                    mapper, demapper, awgn, bit_bank, batch, rounds,
                    label=""):
    """Run decoder at fixed Eb/N0. Returns per-round BLER list and summary dict."""
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    round_blers = []
    n_ack_total = 0
    n_err_total = 0
    total_blocks = batch * rounds

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
        n_ack = int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        n_err = int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())

        n_nack = batch - n_ack
        round_bler = n_nack / batch
        round_blers.append(round_bler)
        n_ack_total += n_ack
        n_err_total += n_err

        print(f"  round {r + 1:3d}/{rounds}  [{label}]  "
              f"BLER={round_bler:.4f}  (ACK {n_ack}/{batch})")

    n_nack_total = total_blocks - n_ack_total
    pooled_bler, ci_lo, ci_hi = wilson_ci(n_nack_total, total_blocks)
    pooled_ber = n_err_total / (total_blocks * K_PAYLOAD)
    round_arr = np.array(round_blers)

    summary = {
        "label": label,
        "ebno_db": ebno,
        "batch": batch,
        "rounds": rounds,
        "n_blocks": total_blocks,
        "n_acks": n_ack_total,
        "n_nacks": n_nack_total,
        "pooled_bler": pooled_bler,
        "pooled_ber": pooled_ber,
        "bler_ci95_lo": ci_lo,
        "bler_ci95_hi": ci_hi,
        "round_bler_mean": float(round_arr.mean()),
        "round_bler_std": float(round_arr.std(ddof=1)) if rounds > 1 else 0.0,
        "round_bler_min": float(round_arr.min()),
        "round_bler_max": float(round_arr.max()),
        "round_blers": round_blers,
    }
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Eb/N0=0.8 BLER 통계 (plot_comparison_poster.py 동일 설정).")
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
    p.add_argument("--ebno", type=float, default=DEFAULT_EBNO)
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--out", default="results/bler_stats_ebno08.json",
                   help="JSON 결과 저장 경로")
    p.add_argument("--csv", default="results/bler_stats_ebno08.csv",
                   help="CSV 요약 저장 경로")
    return p.parse_args()


def print_summary(s: dict):
    print(f"\n  ── {s['label']} 요약 ──")
    print(f"  총 블록 수     : {s['n_blocks']}  (batch={s['batch']} × rounds={s['rounds']})")
    print(f"  ACK / NACK     : {s['n_acks']} / {s['n_nacks']}")
    print(f"  Pooled BLER    : {s['pooled_bler']:.6f}")
    print(f"  95% CI (Wilson): [{s['bler_ci95_lo']:.6f}, {s['bler_ci95_hi']:.6f}]")
    print(f"  Pooled BER     : {s['pooled_ber']:.4e}")
    print(f"  Round BLER mean±std : {s['round_bler_mean']:.6f} ± {s['round_bler_std']:.6f}")
    print(f"  Round BLER min/max  : {s['round_bler_min']:.6f} / {s['round_bler_max']:.6f}")


def main():
    args = parse_args()

    if args.alpha is not None:
        n_calls = max(len(args.bp_schedule) - 1, 1)
        alpha_sched = [args.alpha] * n_calls
    else:
        alpha_sched = parse_alpha_schedule(args.alpha_schedule)

    bp_total = sum(args.bp_schedule)

    print("=" * 68)
    print("stats_bler_ebno08")
    print(f"  Eb/N0          : {args.ebno} dB")
    print(f"  Baselines      : {['BP-' + str(n) for n in args.baselines]}")
    print(f"  BP schedule    : {args.bp_schedule}  ({bp_total} total iter)")
    print(f"  alpha schedule : {fmt_sched(alpha_sched)}")
    print(f"  beta           : {args.beta}")
    print(f"  sigma          : {args.sigma}")
    print(f"  batch={args.batch}  rounds={args.rounds}")
    print("=" * 68)

    crc_enc  = CRCEncoder(CRC_DEGREE)
    crc_dec  = CRCDecoder(crc_enc)
    k_ldpc   = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper   = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn     = AWGN()
    bit_bank = build_test_bitbank()

    baseline_decoders = {}
    for n_iter in sorted(set(args.baselines)):
        baseline_decoders[n_iter] = LDPC5GDecoder(
            ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
            cn_schedule="flooding", hard_out=False, return_infobits=True,
            num_iter=n_iter, llr_max=30.0)

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

    all_summaries = []

    for n_iter in sorted(set(args.baselines)):
        label = f"BP-{n_iter}"
        print(f"\n--- Baseline {label} @ Eb/N0={args.ebno} dB ---")
        s = eval_bler_stats(
            baseline_decoders[n_iter], args.ebno,
            ldpc_enc, crc_enc, crc_dec,
            mapper, demapper, awgn, bit_bank,
            args.batch, args.rounds, label=label)
        print_summary(s)
        all_summaries.append(s)

    print(f"\n--- Proposed {args.bp_schedule} @ Eb/N0={args.ebno} dB ---")
    s_prop = eval_bler_stats(
        dec_dn, args.ebno, ldpc_enc, crc_enc, crc_dec,
        mapper, demapper, awgn, bit_bank,
        args.batch, args.rounds, label="Proposed")
    print_summary(s_prop)
    all_summaries.append(s_prop)

    # 비교 테이블
    print("\n" + "=" * 68)
    print(f"Eb/N0 = {args.ebno} dB  BLER 비교 (총 {args.batch * args.rounds} blocks each)")
    print("=" * 68)
    print(f"{'Decoder':<16} {'Pooled BLER':>12} {'95% CI':>28} {'Round mean±std':>22}")
    print("-" * 68)
    for s in all_summaries:
        ci = f"[{s['bler_ci95_lo']:.4f}, {s['bler_ci95_hi']:.4f}]"
        rs = f"{s['round_bler_mean']:.4f}±{s['round_bler_std']:.4f}"
        print(f"{s['label']:<16} {s['pooled_bler']:>12.6f} {ci:>28} {rs:>22}")

    # 저장
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_data = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": {
            "ebno_db": args.ebno,
            "bp_schedule": args.bp_schedule,
            "alpha_schedule": alpha_sched,
            "beta": args.beta,
            "sigma": args.sigma,
            "batch": args.batch,
            "rounds": args.rounds,
            "baselines": args.baselines,
        },
        "results": [{k: v for k, v in s.items() if k != "round_blers"} for s in all_summaries],
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nJSON saved → {args.out}")

    csv_fields = [
        "label", "ebno_db", "n_blocks", "n_acks", "n_nacks",
        "pooled_bler", "bler_ci95_lo", "bler_ci95_hi", "pooled_ber",
        "round_bler_mean", "round_bler_std", "round_bler_min", "round_bler_max",
    ]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_summaries)
    print(f"CSV saved  → {args.csv}")


if __name__ == "__main__":
    main()
