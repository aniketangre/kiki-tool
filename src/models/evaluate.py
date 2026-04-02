# models/evaluate.py
# Evaluates a trained model on test data and reports performance metrics.
#
# Two metrics are computed:
#   - MAE (Mean Absolute Error) for rotation angle predictions (regression)
#   - Accuracy for cell type prediction (classification via argmax)

import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import mean_absolute_error, accuracy_score

from data.load_data import load_data
from data.preprocess import preprocess
from utils.utils import normalize, get_dataset_partitions, denormalize


def evaluate_model(model, x_test, y_test, x_normalizer, y_normalizer):
    """
    Evaluate the trained model on test data.

    The model predicts 19 outputs:
      - Outputs 0-2  : rotation angles (x_rot, y_rot, z_rot) → evaluated with MAE
      - Outputs 3-18 : cell type as one-hot vector            → evaluated with accuracy

    Parameters:
        model: Trained scikit-learn model.
        x_test (np.ndarray): Normalized input test data (1D or 2D).
        y_test (np.ndarray): Normalized output test data (1D or 2D).
        x_normalizer (np.ndarray): Normalizer for inputs, shape (n_inputs, 2).
        y_normalizer (np.ndarray): Normalizer for outputs, shape (n_outputs, 2).

    Returns:
        mae (float): Mean absolute error for rotation angles (in degrees).
        cell_accuracy (float): Accuracy for cell type prediction (0–100%).
    """
    if x_test.ndim == 1:
        # --- Single sample ---
        print("Predicting output for one input sample.")
        y_true = denormalize(y_test, y_normalizer)
        y_pred_norm = model.predict(np.expand_dims(x_test, axis=0))[0]
        y_pred = denormalize(y_pred_norm, y_normalizer)

        mae = mean_absolute_error(y_true[:3], y_pred[:3])
        # argmax finds the index of the highest value — this is the predicted cell type
        cell_accuracy = accuracy_score(
            [np.argmax(y_true[3:])],
            [np.argmax(y_pred[3:])]
        ) * 100

    elif x_test.ndim == 2:
        # --- Multiple samples ---
        print(f"Predicting output for {x_test.shape[0]} input samples.")
        y_true = denormalize(y_test, y_normalizer)
        y_pred_norm = model.predict(x_test)
        y_pred = denormalize(y_pred_norm, y_normalizer)

        mae = mean_absolute_error(y_true[:, :3], y_pred[:, :3])
        cell_accuracy = accuracy_score(
            np.argmax(y_true[:, 3:], axis=1),
            np.argmax(y_pred[:, 3:], axis=1)
        ) * 100

    else:
        print("Error: x_test must be 1D (single sample) or 2D (multiple samples).")
        return None, None

    return mae, cell_accuracy


# -----------------------------------------------------------------------
# Run this file to evaluate the saved model:
#   python src/models/evaluate.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    database_filepath = os.path.join(project_root, "database_upd.xlsx")
    model_path = os.path.join(project_root, "src", "serving", "model", "extra_trees_model.pkl")

    # --- Load and prepare data ---
    print("Loading and preprocessing data...")
    df = load_data(database_filepath)
    df_processed, input_columns, output_columns = preprocess(df, all_params=False)
    normalized_df, normalizer_df = normalize(df_processed)
    train_df, val_df, test_df = get_dataset_partitions(
        normalized_df, train_split=0.9, val_split=0.0, test_split=0.1
    )

    x_test = test_df[input_columns].to_numpy()
    y_test = test_df[output_columns].to_numpy()
    x_normalizer = normalizer_df.loc[input_columns].to_numpy()
    y_normalizer = normalizer_df.loc[output_columns].to_numpy()

    # --- Load the saved model ---
    print(f"\nLoading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # --- Evaluate ---
    mae, cell_acc = evaluate_model(model, x_test, y_test, x_normalizer, y_normalizer)
    print(f"\nOrientation prediction error : +/- {mae:.4f} degrees")
    print(f"Cell type prediction accuracy: {cell_acc:.2f}%")
