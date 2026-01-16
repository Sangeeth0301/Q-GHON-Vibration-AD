import os
import sys
import numpy as np
import torch
import deepxde as dde
from pathlib import Path
from scipy.signal import butter, filtfilt

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from data_prep.load_xjtu import load_xjtu_sy_folder
from data_prep.integrator import integrate_and_scale

def discovery_pinn(t_data, x_data, v_data, a_data):
    """
    Step 1D Optimal: Multi-State Fusion PINN.
    Identifies physical parameters with positivity constraints.
    """
    # 1. Trainable Variables (Starting with a guess of 1.0)
    _k_h = dde.Variable(1.0)
    _k_v = dde.Variable(1.0)
    _k_c = dde.Variable(0.1) 
    _c_d = dde.Variable(0.1) 

    def multi_bearing_system(t, y):
        """
        y contains 6 outputs from the NN: [xh, xv, vh, vv, ah, av]
        """
        xh, xv = y[:, 0:1], y[:, 1:2]
        vh, vv = y[:, 2:3], y[:, 3:4]
        ah, av = y[:, 4:5], y[:, 5:6]

        # RECTIFICATION: Force Physical Positivity
        kh, kv, cd = torch.abs(_k_h), torch.abs(_k_v), torch.abs(_c_d)
        kc = _k_c # Coupling can be +/-

        # Kinematic Links (First-order derivatives are more stable than Hessians)
        dxh_dt = dde.grad.jacobian(y, t, i=0, j=0)
        dxv_dt = dde.grad.jacobian(y, t, i=1, j=0)
        dvh_dt = dde.grad.jacobian(y, t, i=2, j=0)
        dvv_dt = dde.grad.jacobian(y, t, i=3, j=0)

        # Equations to satisfy
        eq_link_vh = dxh_dt - vh    # Velocity is derivative of displacement
        eq_link_vv = dxv_dt - vv
        eq_link_ah = dvh_dt - ah    # Acceleration is derivative of velocity
        eq_link_av = dvv_dt - av
        
        # The Physical ODE (M=1)
        eq_phys_h = ah + cd*vh + kh*xh + kc*xv
        eq_phys_v = av + cd*vv + kv*xv + kc*xh

        return [eq_link_vh, eq_link_vv, eq_link_ah, eq_link_av, eq_phys_h, eq_phys_v]

    geom = dde.geometry.TimeDomain(t_data[0], t_data[-1])

    # 3. Observations (Directly supervising all 6 states)
    obs_xh = dde.icbc.PointSetBC(t_data, x_data[:, 0:1], component=0)
    obs_xv = dde.icbc.PointSetBC(t_data, x_data[:, 1:2], component=1)
    obs_vh = dde.icbc.PointSetBC(t_data, v_data[:, 0:1], component=2)
    obs_vv = dde.icbc.PointSetBC(t_data, v_data[:, 1:2], component=3)
    obs_ah = dde.icbc.PointSetBC(t_data, a_data[:, 0:1], component=4)
    obs_av = dde.icbc.PointSetBC(t_data, a_data[:, 1:2], component=5)

    data = dde.data.PDE(
        geom,
        multi_bearing_system,
        [obs_xh, obs_xv, obs_vh, obs_vv, obs_ah, obs_av],
        num_domain=500,
        anchors=t_data,
    )

    # 4. Neural Network (1 input t -> 6 outputs)
    net = dde.nn.FNN([1] + [64] * 3 + [6], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # 5. Loss Weighting (Data observations are weighted 100x higher than PDE)
    # [4 links + 2 physics + 6 observations]
    loss_weights = [1, 1, 1, 1, 2, 2, 100, 100, 100, 100, 100, 100]
    
    variable_list = [_k_h, _k_v, _k_c, _c_d]
    model.compile("adam", lr=0.0005, external_trainable_variables=variable_list, loss_weights=loss_weights)

    print("--- Starting Physics Discovery Training ---")
    model.train(iterations=10000)
    
    return model, variable_list

if __name__ == "__main__":
    DATA_FOLDER = r"C:\Vibration_Data_Master\XJTU_SY\Bearing1_1"
    FS = 25600
    
    # --- PHYSICAL RECTIFICATION: USE BANDPASS SIGNAL, NOT ENVELOPE ---
    def get_bandpassed_vibration(folder, max_files=1):
        raw = load_xjtu_sy_folder(folder, max_files=max_files)
        nyq = 0.5 * FS
        b, a = butter(4, [2000/nyq, 10000/nyq], btype='band')
        return filtfilt(b, a, raw, axis=0)

    print("--- STEP 1 & 2: Loading & Filtering (Oscillatory Signal) ---")
    bandpassed = get_bandpassed_vibration(DATA_FOLDER)
    
    print("--- STEP 3: Reconstructing State-Space ---")
    x_s, v_s, a_s, _ = integrate_and_scale(bandpassed, fs=FS)
    
    n = 2000 # Samples for discovery
    t = np.linspace(0, n/FS, n).reshape(-1, 1).astype(np.float32)
    x_in, v_in, a_in = x_s[:n].astype(np.float32), v_s[:n].astype(np.float32), a_s[:n].astype(np.float32)

    print(f"--- STEP 4: Commencing DeepXDE Discovery on {n} samples ---")
    model, vars = discovery_pinn(t, x_in, v_in, a_in)
    
    # Final Result Extraction
    res = [torch.abs(v).detach().cpu().item() if i != 2 else v.detach().cpu().item() 
           for i, v in enumerate(vars)]
    
    print("\n" + "="*40)
    print("SUCCESS! HEALTHY PHYSICAL PARAMETERS DISCOVERED:")
    print(f"Stiffness H (k_h): {res[0]:.6f}")
    print(f"Stiffness V (k_v): {res[1]:.6f}")
    print(f"Coupling (k_c):    {res[2]:.6f}")
    print(f"Damping (c_d):     {res[3]:.6f}")
    print("="*40)