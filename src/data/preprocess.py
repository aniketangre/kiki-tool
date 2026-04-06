# data/preprocess.py
# Preprocesses the raw dataframe: encodes categorical variables,
# selects columns, normalizes, and splits into train/test sets.

import os
import sys
import pandas as pd

# Allow imports from sibling folders (data, features, utils)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_data
from features.build_features import get_columns
from utils.utils import normalize, get_dataset_partitions


def preprocess(df, all_params=False):
    """
    Transform the raw dataframe into a model-ready format.

    Steps:
      1. One-hot encode the 'cell_type' categorical column.
         e.g. "BCC" → a new column 'BCC' with value 1, all others 0.
      2. Drop columns that are not needed for training.
      3. Reorder columns so inputs come first, outputs come last.

    Parameters:
        df (pd.DataFrame): Raw dataframe from load_data().
        all_params (bool): Whether to use all 16 inputs or just 14.

    Returns:
        df_processed (pd.DataFrame): Preprocessed dataframe.
        input_columns (list): Names of the input columns.
        output_columns (list): Names of the output columns.
    """
    # One-hot encode the 'cell_type' column
    # pd.get_dummies turns one categorical column into multiple binary columns
    cell_type_dummies = pd.get_dummies(df["cell_type"]).astype(dtype=int)
    df = pd.concat([df, cell_type_dummies], axis=1)

    # Drop columns we no longer need
    df.drop(columns=["sr_no", "cell_type"], inplace=True)

    # Get the column order and input/output split
    reordered_columns, input_columns, output_columns = get_columns(all_params)

    # Keep only the required columns in the correct order
    df_processed = df.loc[:, reordered_columns]

    return df_processed, input_columns, output_columns


# -----------------------------------------------------------------------
# Run this file directly to test the full preprocessing pipeline:
#   python src/data/preprocess.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project_root, "data", "raw", "database_upd.xlsx")

    # Load raw data
    df = load_data(filepath)

    # Preprocess
    df_processed, input_columns, output_columns = preprocess(df, all_params=False)
    print(f"\nPreprocessed shape: {df_processed.shape}")
    print(f"Input columns  ({len(input_columns)}): {input_columns}")
    print(f"Output columns ({len(output_columns)}): {output_columns}")

    # Normalize and split
    normalized_df, normalizer_df = normalize(df_processed)
    train_df, val_df, test_df = get_dataset_partitions(
        normalized_df, train_split=0.9, val_split=0.0, test_split=0.1
    )
    print(f"\nTraining set : {train_df.shape}")
    print(f"Test set     : {test_df.shape}")

    # Save the processed files to data/processed/
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    normalizer_df.to_csv(os.path.join(output_dir, "normalizer.csv"))
    train_df.to_csv(os.path.join(output_dir, "norm_train.csv"))
    test_df.to_csv(os.path.join(output_dir, "norm_test.csv"))
    print(f"\nSaved processed files to: {output_dir}")
