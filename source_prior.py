"""
SourcePriorDenoiser — PyTorch adapter that connects an EDM-style score
denoiser to the existing iterative BP pipeline.

[onlyextrinsic variant — independent alpha/beta]
  Always returns pure source extrinsic:
    src_ext = src_post - input_llr
  The decoder controls BP vs source weighting via alpha/beta.
"""

import math
import torch
import torch.nn as nn

from score_denoiser import EDMPrecond


class SourcePriorDenoiser(nn.Module):

    def __init__(self,
                 img_h: int = 28,
                 img_w: int = 28,
                 bits_per_pixel: int = 8,
                 model_channels: int = 64,
                 channel_mult=(1, 2, 2),
                 num_blocks: int = 2,
                 attn_resolutions=(7,),
                 dropout: float = 0.0,
                 sigma_data: float = 0.5,
                 sigma_post: float = 3.0):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.bpp = bits_per_pixel
        self.n_pixels = img_h * img_w
        self.k_payload = self.n_pixels * bits_per_pixel
        self.sigma_post = sigma_post

        weights = torch.tensor(
            [2 ** (bits_per_pixel - 1 - i) for i in range(bits_per_pixel)],
            dtype=torch.float32)
        self.register_buffer('bit_weights', weights)

        masks = torch.zeros(bits_per_pixel, 2 ** bits_per_pixel)
        for i in range(bits_per_pixel):
            w = 2 ** (bits_per_pixel - 1 - i)
            for v in range(2 ** bits_per_pixel):
                masks[i, v] = float((v // w) % 2)
        self.register_buffer('bit_masks', masks)
        self.register_buffer('pixel_values',
                             torch.arange(2 ** bits_per_pixel, dtype=torch.float32))

        img_res = max(img_h, img_w)
        self.net = EDMPrecond(
            img_resolution=img_res,
            img_channels=1,
            sigma_data=sigma_data,
            model_type='SongUNet',
            model_channels=model_channels,
            channel_mult=list(channel_mult),
            channel_mult_emb=4,
            num_blocks=num_blocks,
            attn_resolutions=list(attn_resolutions),
            dropout=dropout,
            embedding_type='positional',
            channel_mult_noise=1,
            encoder_type='standard',
            decoder_type='standard',
            resample_filter=[1, 1],
        )

    # ------------------------------------------------------------------
    # LLR ↔ image conversions
    # ------------------------------------------------------------------

    def llr_to_soft_field(self, llr):
        B = llr.shape[0]
        p = torch.sigmoid(llr)
        p = p.reshape(B, self.n_pixels, self.bpp)
        soft_pixel = (p * self.bit_weights).sum(dim=-1)
        img = soft_pixel / 255.0
        img = img.reshape(B, 1, self.img_h, self.img_w)
        return img

    def soft_field_to_posterior_logits(self, img):
        B = img.shape[0]
        pix = img.reshape(B, self.n_pixels) * 255.0

        diff = pix.unsqueeze(-1) - self.pixel_values
        log_pv = -0.5 * diff ** 2 / (self.sigma_post ** 2)
        p_v = torch.softmax(log_pv, dim=-1)

        p_bit1 = torch.einsum('bpv,iv->bip', p_v, self.bit_masks)
        p_bit1 = p_bit1.clamp(1e-7, 1 - 1e-7)
        llr = torch.log(p_bit1 / (1 - p_bit1))
        llr = llr.permute(0, 2, 1).reshape(B, -1)
        return llr

    # ------------------------------------------------------------------
    # Extrinsic computation  (pure extrinsic, no gamma)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_source_extrinsic(posterior_logits, input_llr):
        """src_ext = src_post - input_llr  (turbo principle)"""
        return posterior_logits - input_llr

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, llr, sigma):
        """
        Returns
        -------
        src_extrinsic : [B, K]   approximate source extrinsic LLR
        """
        soft_img = self.llr_to_soft_field(llr)

        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], dtype=torch.float32, device=llr.device)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        sigma = sigma.expand(llr.shape[0])

        denoised = self.net(soft_img, sigma)
        denoised = denoised.clamp(0.0, 1.0)

        posterior = self.soft_field_to_posterior_logits(denoised)
        src_ext = self.compute_source_extrinsic(posterior, llr)
        return src_ext
