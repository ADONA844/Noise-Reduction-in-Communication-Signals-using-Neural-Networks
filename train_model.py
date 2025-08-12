import numpy as np

data = np.load("train_data.npz")
X_train, y_train = data["noisy"], data["clean"]

data = np.load("val_data.npz")
X_val, y_val = data["noisy"], data["clean"]

data = np.load("test_data.npz")
X_test, y_test = data["noisy"], data["clean"]