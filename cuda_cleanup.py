"""
cuda_cleanup.py — between-run memory release helper.

Run this between vanilla_sweep.py calls to clear GPU/CPU memory:
  python3 cuda_cleanup.py

Clears:
  - Python gc
  - TensorFlow session / GPU memory (if TF was imported)
  - PyTorch CUDA cache (if torch was imported)

Exits 0 on success, prints a short status line.
"""
import gc
import sys

released = []

gc.collect()
released.append("gc")

try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        released.append(f"torch-cuda(alloc={torch.cuda.memory_allocated()//1024//1024}MB)")
    else:
        released.append("torch-cpu")
except Exception as e:
    released.append(f"torch-skip({e})")

try:
    import tensorflow as tf
    tf.keras.backend.clear_session()
    released.append("tf-session")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        released.append(f"tf-gpu({len(gpus)})")
except Exception as e:
    released.append(f"tf-skip({e})")

gc.collect()

print(f"[cuda_cleanup] OK  released={released}", flush=True)
sys.exit(0)
