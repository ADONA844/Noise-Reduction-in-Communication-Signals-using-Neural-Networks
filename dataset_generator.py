# dataset_generator.py
"""
Generates BPSK clean/noisy pairs and saves them as .npy
Supports AWGN and Rayleigh fading scenarios.

Usage example:
python dataset_generator.py --out_dir ./data --n_samples 10000 --L 1024 --snr_min -5 --snr_max 20 --channel both
"""

import os
import json
import argparse
import numpy as np
from tqdm import trange

def generate_bpsk(bits):
    # map 0->-1, 1->+1
    return 2*bits - 1

def add_awgn(signal, snr_db):
    # signal shape (..., L) real-valued
    Ps = np.mean(signal**2, axis=-1)  # shape (N,)
    noise_power = Ps / (10**(snr_db/10.0))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, 1, signal.shape) * noise_std[..., None]
    return signal + noise

def apply_rayleigh(signal):
    # per-sample Rayleigh fading amplitude (real positive)
    # scale parameter sigma=1 (standard Rayleigh). Normalize E[a^2]=2*sigma^2 -> here we simply use numpy.rayleigh
    N = signal.shape[0]
    a = np.random.rayleigh(scale=1.0, size=(N,1))
    faded = signal * a
    return faded, a

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    N = args.n_samples
    L = args.L
    snr_min, snr_max = args.snr_min, args.snr_max
    channel = args.channel.lower()

    # we'll generate in-memory if N*L small, else do chunked generation
    X = np.zeros((N, L), dtype=np.float32)  # noisy
    Y = np.zeros((N, L), dtype=np.float32)  # clean
    snrs = np.zeros((N,), dtype=np.float32)
    channel_gains = None
    if channel == 'rayleigh' or channel == 'both':
        channel_gains = np.zeros((N,), dtype=np.float32)

    batch = args.batch_size
    idx = 0
    for i in trange(N):
        bits = rng.integers(0, 2, size=(L,))
        clean = generate_bpsk(bits).astype(np.float32)
        # pick SNR for this sample uniformly in the range
        snr_db = rng.uniform(snr_min, snr_max)
        noisy = None
        if channel == 'awgn':
            noisy = add_awgn(clean[None, :], snr_db)[0]
        elif channel == 'rayleigh':
            faded, a = apply_rayleigh(clean[None, :])
            noisy = add_awgn(faded, snr_db)[0]
            channel_gains[i] = float(a[0,0])
        elif channel == 'both':
            # randomly choose whether sample is awgn-only or rayleigh
            if rng.random() < 0.5:
                noisy = add_awgn(clean[None, :], snr_db)[0]
                channel_gains[i] = 1.0
            else:
                faded, a = apply_rayleigh(clean[None, :])
                noisy = add_awgn(faded, snr_db)[0]
                channel_gains[i] = float(a[0,0])
        else:
            raise ValueError("channel must be 'awgn', 'rayleigh', or 'both'")

        X[i] = noisy
        Y[i] = clean
        snrs[i] = snr_db

    # split
    perm = np.arange(N)
    rng.shuffle(perm)
    train_end = int(N * 0.8)
    val_end = int(N * 0.9)

    idxs_train = perm[:train_end]
    idxs_val = perm[train_end:val_end]
    idxs_test = perm[val_end:]

    np.save(os.path.join(args.out_dir, "train_X.npy"), X[idxs_train])
    np.save(os.path.join(args.out_dir, "train_Y.npy"), Y[idxs_train])
    np.save(os.path.join(args.out_dir, "val_X.npy"), X[idxs_val])
    np.save(os.path.join(args.out_dir, "val_Y.npy"), Y[idxs_val])
    np.save(os.path.join(args.out_dir, "test_X.npy"), X[idxs_test])
    np.save(os.path.join(args.out_dir, "test_Y.npy"), Y[idxs_test])
    np.save(os.path.join(args.out_dir, "snr_all.npy"), snrs)

    metadata = {
        "n_samples": int(N),
        "L": int(L),
        "snr_min": snr_min,
        "snr_max": snr_max,
        "channel": channel,
        "seed": int(args.seed),
    }
    if channel_gains is not None:
        np.save(os.path.join(args.out_dir, "channel_gains.npy"), channel_gains)

    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved data under", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./data")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--snr_min", type=float, default=-5.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--channel", type=str, default="both")
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    main(args)