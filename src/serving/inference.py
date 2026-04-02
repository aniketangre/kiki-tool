# serving/inference.py
# Loads the trained model and runs predictions on new input data.
#
# This module is the bridge between raw user inputs and model predictions.
# It handles: loading the model, normalizing inputs, running the model,
# and converting outputs back to human-readable values.

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import denormalize

# --- File paths (relative to this file) ---
MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH     = os.path.join(MODEL_DIR, "extra_trees_model.pkl")
NORMALIZER_PATH = os.path.join(MODEL_DIR, "normalizer.csv")

# --- Column definitions (must match training exactly) ---
INPUT_COLUMNS = [
    "body_wt (kg)",
    "mat_11 (MPa)", "mat_12 (MPa)", "mat_13 (MPa)",
    "mat_22 (MPa)", "mat_23 (MPa)", "mat_33 (MPa)",
    "mat_44 (MPa)", "mat_55 (MPa)", "mat_66 (MPa)",
    "max_disp (mm)", "max_stress (MPa)",
    "vol_frac", "vox_to_surf (mm)",
]

OUTPUT_COLUMNS = [
    "x_rot (deg)", "y_rot (deg)", "z_rot (deg)",
    "BCC", "DTPMS", "FCC", "FLU", "GYR", "KEV", "OCT",
    "SC", "SCH", "SPP", "DIA", "HCG", "TEG3", "PSM2", "RDO", "DDK2",
]

# Cell type labels in the same order as the one-hot encoding
CELL_TYPES = ["BCC", "DTPMS", "FCC", "FLU", "GYR", "KEV", "OCT",
              "SC", "SCH", "SPP", "DIA", "HCG", "TEG3", "PSM2", "RDO", "DDK2"]


def load_model(model_path=MODEL_PATH):
    """
    Load the trained model from disk.

    Raises FileNotFoundError if the model has not been trained yet.
    To train: run  python src/models/train.py
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please train the model first by running: python src/models/train.py"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_normalizer(normalizer_path=NORMALIZER_PATH):
    """
    Load the normalization parameters (min/max per column) saved during training.

    These are needed to scale incoming inputs and to reverse-scale the outputs.
    """
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"Normalizer file not found at: {normalizer_path}\n"
            "Please train the model first by running: python src/models/train.py"
        )
    normalizer_df = pd.read_csv(normalizer_path, index_col=0)
    return normalizer_df


def normalize_input(input_values, normalizer_df):
    """
    Scale a raw input array to [0, 1] using the training-time min/max values.

    Parameters:
        input_values (list or np.ndarray): 14 raw input values.
        normalizer_df (pd.DataFrame): Loaded normalizer with 'max' and 'min' columns.

    Returns:
        np.ndarray: Normalized input array, ready for the model.
    """
    input_array  = np.array(input_values, dtype=float)
    x_normalizer = normalizer_df.loc[INPUT_COLUMNS].to_numpy()  # shape (14, 2)

    # Min-Max formula: (x - min) / (max - min)
    normalized = (input_array - x_normalizer[:, 1]) / (x_normalizer[:, 0] - x_normalizer[:, 1])
    return normalized


def predict(input_values, model=None, normalizer_df=None):
    """
    Run a full prediction for one set of input values.

    Parameters:
        input_values (list): 14 input values in the order of INPUT_COLUMNS.
        model: Pre-loaded model object. If None, loads from disk.
        normalizer_df: Pre-loaded normalizer DataFrame. If None, loads from disk.

    Returns:
        dict: {
            "x_rotation_deg": float,
            "y_rotation_deg": float,
            "z_rotation_deg": float,
            "recommended_cell_type": str
        }
    """
    # Load from disk if not already provided
    if model is None:
        model = load_model()
    if normalizer_df is None:
        normalizer_df = load_normalizer()

    # Normalize input
    x_normalized = normalize_input(input_values, normalizer_df)

    # Run prediction — model expects shape (1, 14)
    y_normalized = model.predict(np.expand_dims(x_normalized, axis=0))[0]

    # Denormalize output back to original scale
    y_normalizer = normalizer_df.loc[OUTPUT_COLUMNS].to_numpy()  # shape (19, 2)
    y_pred = denormalize(y_normalized, y_normalizer)

    # First 3 outputs are rotation angles
    x_rot, y_rot, z_rot = float(y_pred[0]), float(y_pred[1]), float(y_pred[2])

    # Remaining 16 outputs are one-hot cell type scores → pick the highest
    cell_type_scores = y_pred[3:]
    recommended_cell_type = CELL_TYPES[np.argmax(cell_type_scores)]

    return {
        "x_rotation_deg": round(x_rot, 4),
        "y_rotation_deg": round(y_rot, 4),
        "z_rotation_deg": round(z_rot, 4),
        "recommended_cell_type": recommended_cell_type,
    }


# -----------------------------------------------------------------------
# Run this file to test inference with example inputs:
#   python src/serving/inference.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Example input values — replace with real measurements
    example_input = [
        70.0,    # body_wt (kg)
        1000.0,  # mat_11 (MPa)
        300.0,   # mat_12 (MPa)
        300.0,   # mat_13 (MPa)
        1000.0,  # mat_22 (MPa)
        300.0,   # mat_23 (MPa)
        1000.0,  # mat_33 (MPa)
        350.0,   # mat_44 (MPa)
        350.0,   # mat_55 (MPa)
        350.0,   # mat_66 (MPa)
        2.0,     # max_disp (mm)
        150.0,   # max_stress (MPa)
        0.3,     # vol_frac
        0.5,     # vox_to_surf (mm)
    ]

    print("Running inference with example inputs...")
    result = predict(example_input)

    print("\nPrediction result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
