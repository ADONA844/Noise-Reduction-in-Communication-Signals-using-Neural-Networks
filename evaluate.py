# evaluate.py
"""
Evaluate a saved model checkpoint on the test set and generate plots.

Example:
python evaluate.py --data_dir ./data --model_path ./models/best_dae.pt --model dae --seq_len 1024 --out_dir ./results
"""

import argparse
import os
import numpy as np
import torch
from data_loader import get_dataloaders
from utils import load_checkpoint, compute_snr_in_db, compute_snr_db, compute_ber, plot_example
from models_dae import DenoisingAutoencoder
from models_cnn import ConvAE1D
from models_lstm import LstmDenoiser

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
    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=2)
    model = get_model(args.model, args.seq_len, device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    snr_in_list = []
    snr_out_list = []
    ber_list = []

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            p_np = preds.cpu().numpy()
            snr_in = compute_snr_in_db(y_np, x_np)
            snr_out = compute_snr_db(y_np, p_np)
            ber = compute_ber(y_np, p_np)
            snr_in_list.append(snr_in)
            snr_out_list.append(snr_out)
            ber_list.append(ber)

            # save example plots for first few batches
            if i < args.plot_examples:
                for j in range(min(x_np.shape[0], 3)):
                    outp = os.path.join(args.out_dir, f"example_batch{i}_sample{j}.png")
                    plot_example(y_np[j], x_np[j], p_np[j], out_path=outp)

    snr_in_all = np.concatenate(snr_in_list, axis=0)
    snr_out_all = np.concatenate(snr_out_list, axis=0)
    ber_all = np.concatenate(ber_list, axis=0)

    print("Test samples:", snr_in_all.shape[0])
    print(f"SNR_in mean: {snr_in_all.mean():.3f} dB  SNR_out mean: {snr_out_all.mean():.3f} dB ΔSNR: {snr_out_all.mean() - snr_in_all.mean():.3f} dB")
    print(f"BER mean: {ber_all.mean():.6f}  Accuracy: {1-ber_all.mean():.4f}")

    np.save(os.path.join(args.out_dir, "snr_in_all.npy"), snr_in_all)
    np.save(os.path.join(args.out_dir, "snr_out_all.npy"), snr_out_all)
    np.save(os.path.join(args.out_dir, "ber_all.npy"), ber_all)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model", type=str, default="dae", choices=["dae","cnn","lstm"])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="./results/plots")
    ap.add_argument("--plot_examples", type=int, default=3)
    args = ap.parse_args()
    main(args)