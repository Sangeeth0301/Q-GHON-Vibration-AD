from scipy.signal import butter, filtfilt, hilbert
import numpy as np
import matplotlib.pyplot as plt
from load_xjtu import load_xjtu_sy_folder # Ensure this filename is correct

def apply_hilbert_envelope(signal, fs=25600, lowcut=2000, highcut=10000):
    """
    Step 1B: Bandpass Filter + Hilbert Transform + Smoothing.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # 1. Design Butterworth Bandpass Filter (Order 4)
    b, a = butter(4, [low, high], btype='band')
    
    # 2. Apply filter to both channels (Zero-phase distortion)
    # This is where 'filt_h' and 'filt_v' are defined!
    filt_h = filtfilt(b, a, signal[:, 0])
    filt_v = filtfilt(b, a, signal[:, 1])
    
    # 3. Hilbert Transform to get Magnitude (Envelope)
    env_h = np.abs(hilbert(filt_h))
    env_v = np.abs(hilbert(filt_v))
    
    # 4. Smoothing: Moving Average (Optional but better for PINNs)
    # Helps the model focus on the 'Impact Trend' rather than micro-noise
    window_size = 10
    kernel = np.ones(window_size) / window_size
    env_h = np.convolve(env_h, kernel, mode='same')
    env_v = np.convolve(env_v, kernel, mode='same')
    
    # Stack back into (N, 2)
    envelope_signal = np.vstack([env_h, env_v]).T
    
    return envelope_signal

if __name__ == "__main__":
    # 1. Load Step 1A Data
    DATA_FOLDER = r"C:\Vibration_Data_Master\XJTU_SY\Bearing1_1"
    accel = load_xjtu_sy_folder(DATA_FOLDER, max_files=5)

    # 2. Process Step 1B
    envelope = apply_hilbert_envelope(accel, fs=25600)

    print("Step 1B Success. Envelope shape:", envelope.shape)

    # 3. Visualization (Journal Result)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(accel[:2000, 0], label="Raw Acceleration")
    plt.title("XJTU-SY Bearing 1-1: Raw vs. Cleaned Envelope")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(envelope[:2000, 0], color='orange', label="Hilbert Envelope (Smoothed)")
    plt.xlabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.show()