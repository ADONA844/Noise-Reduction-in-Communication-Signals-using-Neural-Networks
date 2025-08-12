import numpy as np

def generate_signal(length=1024):
    t = np.linspace(0, 1, length)
    # Example clean signal: sine + cosine
    return np.sin(2*np.pi*5*t) + 0.5*np.cos(2*np.pi*10*t)

def add_awgn(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def create_dataset(num_samples, length=1024, snr_range=(0, 20)):
    noisy_signals = []
    clean_signals = []
    for _ in range(num_samples):
        clean = generate_signal(length)
        snr = np.random.uniform(*snr_range)
        noisy = add_awgn(clean, snr)
        noisy_signals.append(noisy)
        clean_signals.append(clean)
    return np.array(noisy_signals), np.array(clean_signals)

if __name__ == "__main__":
    # Dataset sizes
    train_noisy, train_clean = create_dataset(8000)
    val_noisy, val_clean = create_dataset(1000)
    test_noisy, test_clean = create_dataset(1000)

    np.savez("train_data.npz", noisy=train_noisy, clean=train_clean)
    np.savez("val_data.npz", noisy=val_noisy, clean=val_clean)
    np.savez("test_data.npz", noisy=test_noisy, clean=test_clean)

    print("Datasets generated and saved.")