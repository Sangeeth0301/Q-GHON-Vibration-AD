import torch
import torch.nn as nn

class SpectralEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(SpectralEncoder, self).__init__()
        """
        Week 2: Temporal Energy Encoder
        Maps (6, 1024) Hamiltonian Snapshots to a local Spectral Manifold.
        """
        # Block 1: Shallow Feature Extraction (Local transients)
        self.layer1 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Block 2: Deep Spectral Projection (Resonance modes)
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Block 3: Global Feature Aggregation
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Boils 1024 points down to 1 feature per channel
        )
        
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        # x shape: (Batch, 6, 1024)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten to (Batch, 64)
        z = self.fc(x)
        return z

if __name__ == "__main__":
    # Sanity Check
    model = SpectralEncoder(latent_dim=64)
    dummy_input = torch.randn(16, 6, 1024) # 16 windows of Hamiltonian snapshots
    latent_z = model(dummy_input)
    
    print(f"Input Snapshot Shape: {dummy_input.shape}")
    print(f"Latent Spectral Manifold Shape: {latent_z.shape}") # Should be (16, 64)