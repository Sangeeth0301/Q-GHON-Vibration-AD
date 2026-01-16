import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

def butter_highpass_filter(data, cutoff=20, fs=25600, order=5):
    """
    Mandatory for Journal Quality: Removes numerical drift.
    Zero-phase (filtfilt) ensures x, v, and a are perfectly synced.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def integrate_and_scale(accel_signal, fs=25600):
    """
    Step 1C (Fixed Physics):
    1. Zero-mean the OSCILLATORY signal (Not the envelope).
    2. Numerical Integration with actual dt.
    3. Aggressive Drift Removal (HPF 20Hz).
    4. Z-Score Scaling for PINN stability.
    """
    dt = 1.0 / fs
    
    # --- PHYSICAL RECTIFICATION ---
    # We ensure the signal is centered. Since this is the bandpassed 
    # vibration, it will naturally oscillate around zero.
    a_centered = accel_signal - np.mean(accel_signal, axis=0)
    
    velocity = np.zeros_like(a_centered)
    displacement = np.zeros_like(a_centered)
    
    # --- Part 1: State Reconstruction ---
    for i in range(2): # 0=Horizontal, 1=Vertical
        # First Integration: Accel -> Velocity
        v_raw = cumulative_trapezoid(a_centered[:, i], dx=dt, initial=0)
        velocity[:, i] = butter_highpass_filter(v_raw, cutoff=20, fs=fs)
        
        # Second Integration: Velocity -> Displacement
        x_raw = cumulative_trapezoid(velocity[:, i], dx=dt, initial=0)
        displacement[:, i] = butter_highpass_filter(x_raw, cutoff=20, fs=fs)
    
    # --- Part 2: Standardization ---
    scaler_x, scaler_v, scaler_a = StandardScaler(), StandardScaler(), StandardScaler()
    
    x_s = scaler_x.fit_transform(displacement)
    v_s = scaler_v.fit_transform(velocity)
    a_s = scaler_a.fit_transform(a_centered)
    
    scalers = {'x': scaler_x, 'v': scaler_v, 'a': scaler_a}
    
    return x_s, v_s, a_s, scalers

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
    
    from data_prep.load_xjtu import load_xjtu_sy_folder
    # We need a function that returns the BANDPASSED signal, not just the envelope
    from scipy.signal import butter, filtfilt

    def get_bandpassed_signal(signal, fs=25600):
        nyq = 0.5 * fs
        b, a = butter(4, [2000/nyq, 10000/nyq], btype='band')
        return filtfilt(b, a, signal, axis=0)

    # 1. Pipeline Execution
    DATA_FOLDER = r"C:\Vibration_Data_Master\XJTU_SY\Bearing1_1"
    raw_accel = load_xjtu_sy_folder(DATA_FOLDER, max_files=5)
    
    # PHYSICS FIX: Integrate the BANDPASSED signal, not the envelope
    bandpassed_signal = get_bandpassed_signal(raw_accel, fs=25600)
    
    # Execute Step 1C
    x_s, v_s, a_s, scalers = integrate_and_scale(bandpassed_signal, fs=25600)
    
    # 2. Results Verification
    print(f"Step 1C Success. Samples: {x_s.shape[0]}")
    
    # 3. Save Outputs (Mandatory for Phase 1D)
    os.makedirs("data_prep", exist_ok=True)
    np.save("data_prep/x_s.npy", x_s)
    np.save("data_prep/v_s.npy", v_s)
    np.save("data_prep/a_s.npy", a_s)
    print("Saved states to data_prep/")

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_s[:1000, 0], v_s[:1000, 0])
    plt.title("Horizontal Phase Space (Oscillatory)")
    plt.subplot(1, 2, 2)
    plt.plot(x_s[:1000, 1], v_s[:1000, 1], color='red')
    plt.title("Vertical Phase Space (Oscillatory)")
    plt.show()