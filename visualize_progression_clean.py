"""
visualize_progression_clean.py — Visualizing step-by-step progress of BP + Denoiser decoder.
Creates a clean, poster-ready grid comparing Baseline BP and Proposed progression.
"""
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
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
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft
import torchvision

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP  # 6272
CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1

def llr_to_image_np(llr_tf, k_payload=K_PAYLOAD):
    llr = llr_tf[0, :k_payload].numpy()
    p = 1.0 / (1.0 + np.exp(-np.clip(llr, -20, 20)))
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float32)
    p = p.reshape(-1, BPP)
    pixel = (p * weights).sum(axis=1) / 255.0
    return pixel.reshape(IMG_H, IMG_W)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_idx", type=int, default=7)
    p.add_argument("--ebno", type=float, default=0.55)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.1, help="Scalar alpha fallback")
    p.add_argument("--alpha-schedule", type=str, default="0.10,0.08,0.06,0.04,0.02")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--schedule", type=int, nargs="+", default=[5, 5, 5, 5, 5, 5])
    p.add_argument("--ckpt", default="checkpoints/denoiser.pt")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    schedule = args.schedule

    # Parse alpha schedule
    alpha_sched = [float(x.strip()) for x in args.alpha_schedule.split(",") if x.strip()]

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    # Proposed decoder
    dec_pr = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=schedule,
        alpha=alpha_sched[0], beta=args.beta,
        alpha_schedule=alpha_sched,
        k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(schedule), llr_max=30.0)

    if os.path.isfile(args.ckpt):
        dec_pr.denoiser.load_weights_pt(args.ckpt)
        print(f"[INFO] Loaded Proposed Checkpoint: {args.ckpt}")
    dec_pr.denoiser.sigma = args.sigma

    # Baseline decoder
    dec_bl = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=schedule,
        alpha=0.0, beta=0.0,
        k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(schedule), llr_max=30.0)
    dec_bl.denoiser.sigma = args.sigma

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()

    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)

    idx = args.img_idx % len(ds)
    pil_img, class_id = ds[idx]
    orig_img = np.array(pil_img, dtype=np.float32) / 255.0
    orig_bytes = np.array(pil_img, dtype=np.uint8).reshape(-1)
    orig_bits = np.unpackbits(orig_bytes).astype(np.int32)
    u = tf.constant(orig_bits[None, :], dtype=tf.int32)

    u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
    c = ldpc_enc(u_crc)
    x = mapper(c)
    no = ebnodb2no(args.ebno, NUM_BPS, ldpc_enc.coderate)
    y = awgn(x, no)
    llr_ch = demapper(y, no)

    # ---- step-by-step decoding ----
    llr = tf.reshape(llr_ch, [1, -1])
    if ldpc_enc.num_bits_per_symbol is not None:
        llr = tf.gather(llr, ldpc_enc.out_int_inv, axis=-1)

    # Align padding variables
    llr_5g = tf.concat([tf.zeros([1, 2 * ldpc_enc.z], dec_pr.rdtype), llr], axis=1)
    k_filler = ldpc_enc.k_ldpc - ldpc_enc.k
    nb_punc_bits = (ldpc_enc.n_ldpc - k_filler) - ldpc_enc.n - 2 * ldpc_enc.z
    llr_5g = tf.concat(
        [llr_5g, tf.zeros([1, nb_punc_bits - dec_pr._nb_pruned_nodes], dec_pr.rdtype)],
        axis=1)

    x1_sys = llr_5g[:, :ldpc_enc.k]
    nb_par = ldpc_enc.n_ldpc - k_filler - ldpc_enc.k - dec_pr._nb_pruned_nodes
    x2_par = llr_5g[:, ldpc_enc.k:ldpc_enc.k + nb_par]
    z_short = -tf.cast(dec_pr._llr_max, dec_pr.rdtype) * tf.ones([1, k_filler], dec_pr.rdtype)

    k_payload = min(K_PAYLOAD, int(ldpc_enc.k))
    payload0 = x1_sys[:, :k_payload]
    crc_rest = x1_sys[:, k_payload:]

    # alpha/beta setup
    alpha_vals = [tf.cast(a, dec_pr.rdtype) for a in alpha_sched]
    b = tf.cast(args.beta, dec_pr.rdtype)

    # ============ Row 0: Baseline BPonly 6 steps ============
    # We do NOT pass msg_v2c to prevent compounding numerical differences or potential bugs.
    # We run full 5, 10, 15, 20, 25, 30 iterations from scratch to guarantee strict equivalence to standalone BP.
    baseline_images = []
    bl_intr = payload0
    prev_rs = getattr(dec_bl, "_return_state", False)
    prev_ho = getattr(dec_bl, "_hard_out", False)

    try:
        dec_bl._return_state = True
        dec_bl._hard_out = False
        
        cum_iter = 0
        for iters in schedule:
            cum_iter += iters
            bl_x1 = tf.concat([bl_intr, crc_rest], axis=1)
            bl_llr = tf.concat([bl_x1, z_short, x2_par], axis=1)
            # Run clean decoding with msg_v2c=None for cum_iter iterations
            bl_hat, _ = LDPCBPDecoder.call(
                dec_bl, bl_llr, num_iter=int(cum_iter), msg_v2c=None)
            bl_img = llr_to_image_np(bl_hat, k_payload)
            baseline_images.append(bl_img)
    finally:
        dec_bl._return_state = prev_rs
        dec_bl._hard_out = prev_ho

    # ============ Row 1: Proposed 6 steps (1st BP + 5 source-decoded BP) ============
    proposed_images = []
    payload_intr = payload0
    msg_v2c = None
    prev_rs_pr = getattr(dec_pr, "_return_state", False)
    prev_ho_pr = getattr(dec_pr, "_hard_out", False)

    try:
        dec_pr._return_state = True
        dec_pr._hard_out = False

        for chunk_idx, iters in enumerate(schedule):
            x1_stage = tf.concat([payload_intr, crc_rest], axis=1)
            llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)

            x_hat, msg_v2c = LDPCBPDecoder.call(
                dec_pr, llr_bp, num_iter=int(iters), msg_v2c=msg_v2c)

            bp_img = llr_to_image_np(x_hat, k_payload)
            proposed_images.append(bp_img)

            if chunk_idx < len(schedule) - 1:
                post_payload = x_hat[:, :k_payload]
                bp_ext = post_payload - payload_intr
                src_ext = dec_pr._denoiser(post_payload)
                a_t = alpha_vals[chunk_idx] if chunk_idx < len(alpha_vals) else alpha_vals[-1]
                payload_intr = payload0 + b * bp_ext + a_t * src_ext
    finally:
        dec_pr._return_state = prev_rs_pr
        dec_pr._hard_out = prev_ho_pr

    # Ensure we got exactly 6 images for both baseline and proposed
    assert len(baseline_images) == 6, f"Expected 6 baseline images, got {len(baseline_images)}"
    assert len(proposed_images) == 6, f"Expected 6 proposed images, got {len(proposed_images)}"

    # Create figure with 2x6 grid and no spaces
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))

    # Plot Row 0 (Baseline BPonly)
    for col in range(6):
        ax = axes[0, col]
        ax.imshow(baseline_images[col], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    # Plot Row 1 (Proposed progression)
    for col in range(6):
        ax = axes[1, col]
        ax.imshow(proposed_images[col], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    # Completely remove gaps between subplots
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    os.makedirs("results", exist_ok=True)
    if args.out is None:
        args.out = f"results/progression_clean_idx{args.img_idx}_ebno{args.ebno}.png"
    
    # Save with tight boundaries
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"\nSeamless progression grid saved → {args.out}")

if __name__ == "__main__":
    main()
