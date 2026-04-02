# models/tune.py
# Hyperparameter tuning for the Extra Trees model using Grid Search.
#
# Grid Search systematically tries all combinations of the specified
# hyperparameters and uses cross-validation to find the best combination.
# This takes longer than regular training but can improve model performance.

import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

from data.load_data import load_data
from data.preprocess import preprocess
from utils.utils import normalize, get_dataset_partitions


def tune_extra_trees(x_train, y_train):
    """
    Find the best hyperparameters for ExtraTreesRegressor using Grid Search.

    Hyperparameters being searched:
      - n_estimators    : Number of trees in the forest
      - max_depth       : Maximum depth of each tree (None = grow until pure)
      - min_samples_split: Minimum number of samples needed to split a node

    Parameters:
        x_train (np.ndarray): Normalized training inputs.
        y_train (np.ndarray): Normalized training outputs.

    Returns:
        best_params (dict): Best hyperparameter values found.
        best_model: Trained model using the best parameters.
    """
    # Define the grid of hyperparameter values to try
    param_grid = {
        "n_estimators":     [50, 100, 200],   # More trees = better but slower
        "max_depth":        [20, 50, None],    # None means grow until all leaves are pure
        "min_samples_split": [2, 5],           # Larger values prevent overfitting
    }

    # Base model with a fixed random state for reproducibility
    base_model = ExtraTreesRegressor(random_state=10)

    # GridSearchCV tries every combination and scores with 3-fold cross-validation
    # n_jobs=-1 uses all available CPU cores to speed up the search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,               # 3-fold cross-validation
        scoring="r2",       # R² score (higher is better, max = 1.0)
        n_jobs=-1,          # Use all CPU cores
        verbose=2           # Print progress
    )

    print("Starting hyperparameter search...")
    print(f"Total combinations to try: {3 * 3 * 2} × 3 folds = {3 * 3 * 2 * 3} fits")
    print("This may take several minutes...\n")

    grid_search.fit(x_train, y_train)

    print(f"\nBest parameters found  : {grid_search.best_params_}")
    print(f"Best cross-val R² score: {grid_search.best_score_:.4f}")

    return grid_search.best_params_, grid_search.best_estimator_


# -----------------------------------------------------------------------
# Run this file to tune the model and save the best version:
#   python src/models/tune.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    database_filepath = os.path.join(project_root, "database_upd.xlsx")
    model_dir = os.path.join(project_root, "src", "serving", "model")

    # --- Load and prepare data ---
    print("Loading and preprocessing data...")
    df = load_data(database_filepath)
    df_processed, input_columns, output_columns = preprocess(df, all_params=False)
    normalized_df, normalizer_df = normalize(df_processed)
    train_df, val_df, test_df = get_dataset_partitions(
        normalized_df, train_split=0.9, val_split=0.0, test_split=0.1
    )

    x_train = train_df[input_columns].to_numpy()
    y_train = train_df[output_columns].to_numpy()

    print(f"Training samples: {x_train.shape[0]}\n")

    # --- Run tuning ---
    best_params, best_model = tune_extra_trees(x_train, y_train)

    # --- Save the tuned model ---
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "extra_trees_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nTuned model saved to: {save_path}")
