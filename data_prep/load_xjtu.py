import numpy as np
import pandas as pd
from pathlib import Path

def load_xjtu_sy(file_path):
    df = pd.read_csv(file_path)


    # FORCE numeric conversion
    accel = df.astype(float).values  # shape (N, 2)

    if accel.shape[1] != 2:
        raise ValueError("Expected 2 channels (Horizontal, Vertical)")

    accel_clean = accel - np.mean(accel, axis=0)

    return accel_clean


def load_xjtu_sy_folder(folder_path, max_files=5):
    """
    Load multiple healthy XJTU-SY CSV files from a folder
    (Bearing1_1.csv to Bearing1_5.csv) and stack them.

    Returns
    -------
    accel_all : np.ndarray, shape (N_total, 2)
    """

    folder_path = Path(folder_path)

    # Sort files to preserve time order
    csv_files = sorted(folder_path.glob("*.csv"))[:max_files]

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in folder")

    accel_list = []

    for csv_file in csv_files:
        accel_clean = load_xjtu_sy(csv_file)
        accel_list.append(accel_clean)

    # Stack along time axis
    accel_all = np.vstack(accel_list)

    return accel_all


if __name__ == "__main__":
    folder = r"C:\Vibration_Data_Master\XJTU_SY\Bearing1_1"

    accel = load_xjtu_sy_folder(folder, max_files=5)

    print("Loaded shape:", accel.shape)
    print("Mean after DC removal:", accel.mean(axis=0))
