"""
SoftDenoiser — TF layer wrapping the PyTorch SourcePriorDenoiser.

[onlyextrinsic variant — independent alpha/beta]
  Always returns pure source extrinsic (src_post - BP_post).
  gamma removed; BP vs source weighting is done in the decoder via alpha/beta.
"""

import numpy as np
import tensorflow as tf
import torch

from source_prior import SourcePriorDenoiser


class SoftDenoiser(tf.keras.layers.Layer):
    """
    Input:  LLR tensor [B, K] — BP posterior for payload bits
    Output: source extrinsic LLR [B, K] = src_post - input_llr
    """

    def __init__(self,
                 img_h: int = 28,
                 img_w: int = 28,
                 bits_per_pixel: int = 8,
                 model_channels: int = 64,
                 channel_mult=(1, 2, 2),
                 num_blocks: int = 2,
                 attn_resolutions=(7,),
                 sigma_data: float = 0.5,
                 sigma_post: float = 3.0,
                 device: str = 'cpu',
                 **kwargs):
        super().__init__(**kwargs)
        self._sigma = 1.0
        self._device = device
        self._prior = SourcePriorDenoiser(
            img_h=img_h,
            img_w=img_w,
            bits_per_pixel=bits_per_pixel,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            sigma_data=sigma_data,
            sigma_post=sigma_post,
        ).to(device)
        self._prior.eval()

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)

    @property
    def prior_model(self):
        return self._prior

    def load_weights_pt(self, path):
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._prior.load_state_dict(state)
        self._prior.eval()

    def save_weights_pt(self, path):
        torch.save(self._prior.state_dict(), path)

    def call(self, llr_tf):
        llr_np = llr_tf.numpy()
        llr_pt = torch.from_numpy(llr_np).float().to(self._device)

        with torch.no_grad():
            ext_pt = self._prior(llr_pt, sigma=self._sigma)

        ext_np = ext_pt.cpu().numpy()
        return tf.constant(ext_np, dtype=llr_tf.dtype)
