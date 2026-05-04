"""
LDPC5GDecoder_soft — BP + source-extrinsic denoiser.

[onlyextrinsic variant — independent alpha/beta, turbo-style extrinsic]

  Turbo-style extrinsic definitions
  ---------------------------------
  At each comma in the schedule, let ``payload_intr`` be the current a priori
  input that was fed into BP for this chunk.  Then:

    bp_ext  = BP_post  - payload_intr     (BP extrinsic = posterior - a priori)
    src_ext = src_post - BP_post          (source extrinsic = posterior - denoiser input)

  The denoiser is applied to BP_post, so ``BP_post`` is exactly the a priori
  input seen by the denoiser, and ``src_ext`` matches the textbook
  ``posterior - a_priori`` form.

  Update equation
  ---------------
    new_input = channel + beta * bp_ext + alpha * src_ext

  Knobs:
    alpha  — source/denoiser extrinsic weight
    beta   — BP extrinsic weight

  Special cases:
    alpha=0, beta=0    →  new_input = channel                (baseline BP, no denoiser)
    alpha=0.1, beta=0  →  new_input = channel + 0.1*src_ext  (denoiser-only correction)
    alpha=0,   beta=1  →  new_input = BP_post                (turbo BP feedback)

  Notes / approximations:
    * In the FIRST chunk, payload_intr == channel, so bp_ext reduces to
      ``BP_post - channel`` (what the previous variant always used).
      For later chunks, ``payload_intr`` already contains the previous
      extrinsic terms, so subtracting it gives a strictly cleaner extrinsic.
    * The denoiser is non-linear, so ``src_post - BP_post`` is the standard
      practical approximation of true source extrinsic in the turbo sense.
"""
from __future__ import annotations

import tensorflow as tf
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder, LDPCBPDecoder

from denoiser import SoftDenoiser


class LDPC5GDecoder_soft(LDPC5GDecoder):

    def __init__(self, encoder, *, bp_schedule=None,
                 alpha=0.0, beta=0.0,
                 k_payload=None, img_h=28, img_w=28, bits_per_pixel=8,
                 denoiser_kwargs=None, **kwargs):
        super().__init__(encoder, **kwargs)
        self._bp_schedule_custom = (
            [int(x) for x in bp_schedule] if bp_schedule else [20]
        )
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._k_payload_custom = int(k_payload) if k_payload is not None else None
        dn_kw = denoiser_kwargs or {}
        self._denoiser = SoftDenoiser(
            img_h=img_h, img_w=img_w, bits_per_pixel=bits_per_pixel,
            **dn_kw)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = float(value)

    @property
    def bp_schedule(self):
        return list(self._bp_schedule_custom)

    @bp_schedule.setter
    def bp_schedule(self, value):
        self._bp_schedule_custom = [int(x) for x in value]

    @property
    def denoiser(self):
        return self._denoiser

    def call(self, llr_ch, num_iter=None, msg_v2c=None):
        llr_ch_shape = llr_ch.get_shape().as_list()
        llr = tf.reshape(llr_ch, [-1, self.encoder.n])
        B = tf.shape(llr)[0]

        if self._encoder.num_bits_per_symbol is not None:
            llr = tf.gather(llr, self._encoder.out_int_inv, axis=-1)

        llr_5g = tf.concat(
            [tf.zeros([B, 2 * self.encoder.z], self.rdtype), llr], axis=1)

        k_filler = self.encoder.k_ldpc - self.encoder.k
        nb_punc_bits = (
            (self.encoder.n_ldpc - k_filler)
            - self.encoder.n
            - 2 * self.encoder.z)
        llr_5g = tf.concat(
            [llr_5g,
             tf.zeros([B, nb_punc_bits - self._nb_pruned_nodes], self.rdtype)],
            axis=1)

        x1_sys = llr_5g[:, :self.encoder.k]
        nb_par_bits = (
            self.encoder.n_ldpc - k_filler
            - self.encoder.k - self._nb_pruned_nodes)
        x2_par = llr_5g[:, self.encoder.k:self.encoder.k + nb_par_bits]
        z_short = (
            -tf.cast(self._llr_max, self.rdtype)
            * tf.ones([B, k_filler], self.rdtype))

        schedule = self._bp_schedule_custom
        k_payload = self._k_payload_custom
        if k_payload is None:
            k_payload = int(self.encoder.k)
        k_payload = min(k_payload, int(self.encoder.k))

        payload0 = x1_sys[:, :k_payload]          # channel intrinsic (frozen)
        crc_and_rest = x1_sys[:, k_payload:]

        payload_intr = payload0
        curr_msg_v2c = msg_v2c

        prev_return_state = getattr(self, "_return_state", False)
        prev_hard_out = getattr(self, "_hard_out", False)
        x_hat = None

        a = tf.cast(self._alpha, self.rdtype)
        b = tf.cast(self._beta, self.rdtype)

        try:
            self._return_state = True
            self._hard_out = False

            for idx, iters in enumerate(schedule):
                x1_stage = tf.concat([payload_intr, crc_and_rest], axis=1)
                llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)

                x_hat, curr_msg_v2c = LDPCBPDecoder.call(
                    self, llr_bp,
                    num_iter=int(iters),
                    msg_v2c=curr_msg_v2c)

                if idx < len(schedule) - 1:
                    post_payload = x_hat[:, :k_payload]

                    # Turbo-style BP extrinsic:
                    #   bp_ext = BP_post - (a priori input that was fed to BP)
                    # ``payload_intr`` here is still the value used as input to
                    # this BP chunk; we subtract it BEFORE updating it below.
                    bp_ext = post_payload - payload_intr

                    # Source extrinsic (denoiser input == BP_post == post_payload):
                    #   src_ext = src_post - BP_post
                    src_ext = self._denoiser(post_payload)

                    # new_input = channel + beta * bp_ext + alpha * src_ext
                    payload_intr = payload0 + b * bp_ext + a * src_ext

        finally:
            self._return_state = prev_return_state
            self._hard_out = prev_hard_out

        if self._return_infobits:
            u_hat_logits = x_hat[:, :self.encoder.k]
            if self._hard_out:
                u_hat = tf.cast(u_hat_logits > 0.0, tf.int32)
            else:
                u_hat = u_hat_logits
            out_shape = llr_ch_shape[:-1] + [self.encoder.k]
            out_shape[0] = -1
            u_hat = tf.reshape(u_hat, out_shape)
            if prev_return_state:
                return u_hat, curr_msg_v2c
            return u_hat

        if prev_return_state:
            return x_hat, curr_msg_v2c
        return x_hat
