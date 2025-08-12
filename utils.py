# utils.py
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt

EPS = 1e-12

def compute_snr_db(clean, recon):
    """
    clean, recon shape (B, L) numpy arrays
    returns snr_in (per sample), snr_out
    """
    # compute SNR_in: for each sample, original signal power / noise power
    signal_power = np.sum(clean**2, axis=-1)
    noise_power_out = np.sum((recon - clean)**2, axis=-1)
    snr_out = 10 * np.log10((signal_power + EPS) / (noise_power_out + EPS))
    return snr_out

def compute_snr_in_db(clean, noisy):
    signal_power = np.sum(clean**2, axis=-1)
    noise_power_in = np.sum((noisy - clean)**2, axis=-1)
    snr_in = 10 * np.log10((signal_power + EPS) / (noise_power_in + EPS))
    return snr_in

def compute_ber(clean, recon):
    """
    For BPSK: clean values are in {-1,+1}. Reconstruct bits via sign threshold at 0.
    Returns BER per sample (proportion of bits wrong).
    """
    # convert to bits
    clean_bits = (clean > 0).astype(np.int32)
    recon_bits = (recon > 0).astype(np.int32)
    ber_per_sample = np.mean(clean_bits != recon_bits, axis=-1)
    return ber_per_sample

def plot_example(clean, noisy, recon, out_path=None, idx=0):
    L = clean.shape[-1]
    t = np.arange(L)
    plt.figure(figsize=(10,6))
    plt.plot(t, clean, label='clean', alpha=0.7)
    plt.plot(t, noisy, label='noisy', alpha=0.6)
    plt.plot(t, recon, label='recon', alpha=0.9)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, device='cpu'):
    import torch
    return torch.load(path, map_location=device)