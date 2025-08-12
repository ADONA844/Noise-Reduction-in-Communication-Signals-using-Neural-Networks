import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BPskDataset(Dataset):
    def __init__(self, X_path, Y_path, memmap=False, transform=None, dtype=np.float32):
        if memmap:
            self.X = np.load(X_path, mmap_mode='r')
            self.Y = np.load(Y_path, mmap_mode='r')
        else:
            self.X = np.load(X_path).astype(dtype)
            self.Y = np.load(Y_path).astype(dtype)
        assert self.X.shape == self.Y.shape
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x, y = self.transform(x, y)
        # return as torch tensors: shape (L,) -> we'll unsqueeze channel dim in collate or model
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def get_dataloaders(data_dir, batch_size=64, num_workers=4, memmap=False):
    train_ds = BPskDataset(f"{data_dir}/train_X.npy", f"{data_dir}/train_Y.npy", memmap=memmap)
    val_ds = BPskDataset(f"{data_dir}/val_X.npy", f"{data_dir}/val_Y.npy", memmap=memmap)
    test_ds = BPskDataset(f"{data_dir}/test_X.npy", f"{data_dir}/test_Y.npy", memmap=memmap)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader