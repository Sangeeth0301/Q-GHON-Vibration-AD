import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

def butter_highpass_filter(data, cutoff=20, fs=25600, order=5):
    """
    Mandatory for Journal Quality: Removes numerical drift.
    Zero-phase (filtfilt) ensures x, v, and a are perfectly synced in time.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def integrate_and_scale(accel_envelope, fs=25600):
    """
    Step 1C:
    1. Zero-mean the envelope (Turning energy into energy fluctuations).
    2. Numerical Integration with actual dt.
    3. Aggressive Drift Removal (HPF 20Hz).
    4. Z-Score Scaling (The PINN Gradient Secret).
    """
    dt = 1.0 / fs
    
    # --- CRITICAL PHYSICS STEP: ZERO-MEAN THE ENVELOPE ---
    # Since envelopes are strictly positive, we subtract the mean to 
    # allow the system to 'vibrate' around zero for the ODE Ma + Cv + Kx = 0
    a_centered = accel_envelope - np.mean(accel_envelope, axis=0)
    
    # Initialize containers
    velocity = np.zeros_like(a_centered)
    displacement = np.zeros_like(a_centered)
    
    # --- Part 1: State Reconstruction (Multivariate) ---
    for i in range(2): # 0=Horizontal, 1=Vertical
        # Accel -> Velocity
        v_raw = cumulative_trapezoid(a_centered[:, i], dx=dt, initial=0)
        velocity[:, i] = butter_highpass_filter(v_raw, cutoff=20, fs=fs)
        
        # Velocity -> Displacement
        x_raw = cumulative_trapezoid(velocity[:, i], dx=dt, initial=0)
        displacement[:, i] = butter_highpass_filter(x_raw, cutoff=20, fs=fs)
    
    # --- Part 2: Standardization (Non-dimensionalization) ---
    # We scale x, v, a so they all live in the range ~[-3, 3].
    # Without this, the DeepXDE Physics Loss will fail to converge.
    scaler_x = StandardScaler()
    scaler_v = StandardScaler()
    scaler_a = StandardScaler()
    
    x_s = scaler_x.fit_transform(displacement)
    v_s = scaler_v.fit_transform(velocity)
    a_s = scaler_a.fit_transform(a_centered)
    
    scalers = {'x': scaler_x, 'v': scaler_v, 'a': scaler_a}
    
    return x_s, v_s, a_s, scalers

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd()) # Ensure project root is in path
    
    # Import your previous logic
    from data_prep.load_xjtu import load_xjtu_sy_folder
    from data_prep.envelope_engine import apply_hilbert_envelope

    # 1. Pipeline Execution (Healthy Data)
    DATA_FOLDER = r"C:\Vibration_Data_Master\XJTU_SY\Bearing1_1"
    print("Executing Step 1C Pipeline...")
    
    raw_accel = load_xjtu_sy_folder(DATA_FOLDER, max_files=5)
    envelope = apply_hilbert_envelope(raw_accel, fs=25600)
    
    # Execute Step 1C
    x_s, v_s, a_s, scalers = integrate_and_scale(envelope, fs=25600)
    
    # 2. Results Verification
    print(f"Step 1C Success. Samples Processed: {x_s.shape[0]}")
    print(f"Displacement Scaled Range: [{x_s.min():.2f}, {x_s.max():.2f}]")

    # 3. Visualization: The State-Space Trajectory (The Baseline)
    plt.figure(figsize=(14, 8))
    
    # Top Plot: Time-Sync Check
    plt.subplot(2, 1, 1)
    plt.plot(a_s[:1000, 0], label="Scaled Accel (a)", alpha=0.6)
    plt.plot(x_s[:1000, 0], label="Scaled Disp (x)", linewidth=2)
    plt.title("Step 1C: Physics-Consistent Scaled States (Healthy)")
    plt.legend()
    plt.grid(True)
    
    # Bottom Left: Horizontal Phase Space
    plt.subplot(2, 2, 3)
    plt.plot(x_s[:10000, 0], v_s[:10000, 0], color='blue', alpha=0.3)
    plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
    plt.title("Horizontal Phase Space (x vs v)")
    plt.xlabel("Scaled Position"); plt.ylabel("Scaled Velocity")
    
    # Bottom Right: Vertical Phase Space
    plt.subplot(2, 2, 4)
    plt.plot(x_s[:10000, 1], v_s[:10000, 1], color='red', alpha=0.3)
    plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
    plt.title("Vertical Phase Space (x vs v)")
    plt.xlabel("Scaled Position"); plt.ylabel("Scaled Velocity")
    
    plt.tight_layout()
    plt.show()