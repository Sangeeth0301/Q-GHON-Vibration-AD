import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

class HamiltonianDataset(Dataset):
    def __init__(self, root_path, window_size=1024, stride=512):
        """
        Step 2A: Physical Data Lifting
        Lifts x_s, v_s, a_s into 6-channel Hamiltonian Snapshots.
        """
        # 1. Load the Physical States from Phase 1C
        path = Path(root_path)
        x_s = np.load(path / "x_s.npy") # Displacement (N, 2)
        v_s = np.load(path / "v_s.npy") # Velocity (N, 2)
        a_s = np.load(path / "a_s.npy") # Acceleration (N, 2)

        # 2A.1: Multichannel Stacking (Preserving Axis Identity)
        # Order: [xh, xv, vh, vv, ah, av]
        # This order is CRITICAL for the GCN and Quantum Mapping later.
        self.data = np.hstack([x_s, v_s, a_s]) 
        
        # 2A.2: Sliding-Window Tensorization
        self.windows = []
        for i in range(0, len(self.data) - window_size, stride):
            window = self.data[i : i + window_size, :] # (1024, 6)
            self.windows.append(window.T) # Transpose to (6, 1024) for CNN
            
        self.windows = np.array(self.windows)
        print(f"Dataset Created: {len(self.windows)} windows of shape (6, 1024)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 2A.3: Axis Identity Preservation
        # Returning as Torch Tensor
        return torch.tensor(self.windows[idx], dtype=torch.float32)

def visualize_snapshot(snapshot):
    """
    Scientific Result: Hamiltonian Snapshot Plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.arange(1024)
    
    # Displacement
    axes[0].plot(t, snapshot[0], label='Disp H')
    axes[0].plot(t, snapshot[1], label='Disp V', alpha=0.7)
    axes[0].set_ylabel('Displacement (q)')
    axes[0].legend(loc='upper right')
    
    # Velocity
    axes[1].plot(t, snapshot[2], label='Vel H', color='green')
    axes[1].plot(t, snapshot[3], label='Vel V', color='lightgreen', alpha=0.7)
    axes[1].set_ylabel('Velocity (p)')
    axes[1].legend(loc='upper right')
    
    # Acceleration
    axes[2].plot(t, snapshot[4], label='Accel H', color='red')
    axes[2].plot(t, snapshot[5], label='Accel V', color='orange', alpha=0.7)
    axes[2].set_ylabel('Acceleration (F/m)')
    axes[2].set_xlabel('Time Steps (L=1024)')
    axes[2].legend(loc='upper right')
    
    plt.suptitle("Step 2A: Hamiltonian Physical Snapshot (6 Channels)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path where you saved your .npy files in Phase 1
    ROOT = "data_prep/" 
    
    # Initialize Dataset
    dataset = HamiltonianDataset(ROOT)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Visualize the first snapshot to verify 2A.3 identity preservation
    sample_batch = next(iter(dataloader))
    print(f"Batch Shape: {sample_batch.shape}") # Should be (16, 6, 1024)
    
    visualize_snapshot(sample_batch[0].numpy())