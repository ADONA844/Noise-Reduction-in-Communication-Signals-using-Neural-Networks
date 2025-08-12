# realtime_test.py
"""
Simulate streaming inference with overlap-add windows and measure latency.
"""

import argparse
import time
import numpy as np
import torch
from models_dae import DenoisingAutoencoder
from models_cnn import ConvAE1D
from models_lstm import LstmDenoiser
from data_loader import BPskDataset
from utils import load_checkpoint

def get_model(name, seq_len, device):
    if name == 'dae':
        return DenoisingAutoencoder(seq_len=seq_len).to(device)
    if name == 'cnn':
        return ConvAE1D(seq_len=seq_len).to(device)
    if name == 'lstm':
        return LstmDenoiser(seq_len=seq_len).to(device)
    raise ValueError

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    ds = BPskDataset(f"{args.data_dir}/test_X.npy", f"{args.data_dir}/test_Y.npy", memmap=True)
    model = get_model(args.model, seq_len=args.seq_len, device=device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # simulate streaming: pick single test sample and slide window across it
    x, y = ds[0]  # torch tensors
    x = x.numpy()
    y = y.numpy()
    L = args.seq_len
    stride = int(L * (1 - args.overlap))
    windows = []
    for start in range(0, len(x)-L+1, stride):
        windows.append(x[start:start+L])
    times = []
    with torch.no_grad():
        for w in windows[:args.max_windows]:
            inp = torch.from_numpy(w).float().unsqueeze(0).to(device)
            t0 = time.perf_counter()
            out = model(inp)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.perf_counter()
            times.append((t1 - t0)*1000.0)
    times = np.array(times)
    print(f"Executed {len(times)} windows. mean latency: {times.mean():.3f} ms, p99: {np.percentile(times,99):.3f} ms")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model", type=str, default="dae", choices=["dae","cnn","lstm"])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--overlap", type=float, default=0.5)  # 50% overlap
    ap.add_argument("--max_windows", type=int, default=200)
    args = ap.parse_args()
    main(args)