# train.py
"""
Unified training script.

Example:
python train.py --data_dir ./data --model dae --seq_len 1024 --epochs 50 --batch_size 64 --lr 1e-3 --device cuda
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_loader import get_dataloaders
from utils import compute_snr_in_db, compute_snr_db, save_checkpoint
from models_dae import DenoisingAutoencoder
from models_cnn import ConvAE1D
from models_lstm import LstmDenoiser
from tqdm import tqdm

def get_model_by_name(name, seq_len, device):
    if name == 'dae':
        return DenoisingAutoencoder(seq_len=seq_len, latent_dim=64).to(device)
    elif name == 'cnn':
        return ConvAE1D(seq_len=seq_len).to(device)
    elif name == 'lstm':
        return LstmDenoiser(seq_len=seq_len).to(device)
    else:
        raise ValueError("unknown model")

def train_epoch(model, loader, opt, criterion, device, clip_grad=None):
    model.train()
    total_loss = 0.0
    cnt = 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        total_loss += loss.item() * x.shape[0]
        cnt += x.shape[0]
    return total_loss / (cnt + 1e-12)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    cnt = 0
    snr_in_list = []
    snr_out_list = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * x.shape[0]
            cnt += x.shape[0]

            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            p_np = preds.cpu().numpy()
            snr_in = compute_snr_in_db(y_np, x_np)
            snr_out = compute_snr_db(y_np, p_np)
            snr_in_list.append(snr_in)
            snr_out_list.append(snr_out)
    if cnt == 0:
        return None
    avg_loss = total_loss / (cnt + 1e-12)
    snr_in_all = np.concatenate(snr_in_list, axis=0)
    snr_out_all = np.concatenate(snr_out_list, axis=0)
    return avg_loss, snr_in_all.mean(), snr_out_all.mean()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, memmap=False)

    model = get_model_by_name(args.model, seq_len=args.seq_len, device=device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = 1e9
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, criterion, device, clip_grad=args.clip_grad)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        if val_metrics is None:
            print("No validation data")
            break
        val_loss, val_snr_in_mean, val_snr_out_mean = val_metrics
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} | val_snr_in={val_snr_in_mean:.3f} dB val_snr_out={val_snr_out_mean:.3f} dB | time={dt:.1f}s")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

    # final test evaluation with best checkpoint
    print("Loading best checkpoint and evaluating test set...")
    ckpt_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    test_loss, test_snr_in_mean, test_snr_out_mean = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.6f} | test_snr_in={test_snr_in_mean:.3f} dB | test_snr_out={test_snr_out_mean:.3f} dB | ΔSNR={test_snr_out_mean - test_snr_in_mean:.3f} dB")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--model", type=str, default="dae", choices=["dae","cnn","lstm"])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--save_dir", type=str, default="./models")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    args = ap.parse_args()
    main(args)