# models/train.py
# Trains three tree-based regression models and saves the best one.
#
# Models trained:
#   1. Decision Tree  - single tree, simple baseline
#   2. Random Forest  - ensemble of trees with bootstrapped samples
#   3. Extra Trees    - ensemble with fully random splits (best performer)

import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from data.load_data import load_data
from data.preprocess import preprocess
from utils.utils import normalize, get_dataset_partitions


def train_decision_tree(x_train, y_train, max_depth=500, random_state=10):
    """
    Train a single Decision Tree Regressor.

    A decision tree learns by splitting the data repeatedly based on
    the feature values, creating a tree of decisions.
    """
    print("  Training Decision Tree...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, max_depth=50, n_estimators=100, random_state=10):
    """
    Train a Random Forest Regressor.

    Random Forest builds many decision trees on random subsets of the data
    (bootstrapping) and averages their predictions. This reduces overfitting
    compared to a single tree.
    """
    print("  Training Random Forest...")
    model = RandomForestRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(x_train, y_train)
    return model


def train_extra_trees(x_train, y_train, max_depth=50, n_estimators=100, random_state=10):
    """
    Train an Extra Trees Regressor (Extremely Randomized Trees).

    Extra Trees is similar to Random Forest but uses the full dataset (no
    bootstrapping) and selects split thresholds completely at random.
    This extra randomness reduces variance and often gives better generalization.
    """
    print("  Training Extra Trees...")
    model = ExtraTreesRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(x_train, y_train)
    return model


def save_model(model, save_path):
    """Save a trained model to disk using pickle format."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {save_path}")


# -----------------------------------------------------------------------
# Run this file to train all models and save the best one:
#   python src/models/train.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Paths ---
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    database_filepath = os.path.join(project_root, "data", "raw", "database_upd.xlsx")
    model_dir = os.path.join(project_root, "src", "serving", "model")

    # --- Load and preprocess data ---
    print("Step 1: Loading data...")
    df = load_data(database_filepath)

    print("Step 2: Preprocessing data...")
    df_processed, input_columns, output_columns = preprocess(df, all_params=False)

    # --- Normalize and split ---
    print("Step 3: Normalizing and splitting data...")
    normalized_df, normalizer_df = normalize(df_processed)
    train_df, val_df, test_df = get_dataset_partitions(
        normalized_df, train_split=0.9, val_split=0.0, test_split=0.1
    )

    x_train = train_df[input_columns].to_numpy()
    y_train = train_df[output_columns].to_numpy()
    x_test  = test_df[input_columns].to_numpy()
    y_test  = test_df[output_columns].to_numpy()

    print(f"  Training samples : {x_train.shape[0]}")
    print(f"  Test samples     : {x_test.shape[0]}")
    print(f"  Input features   : {x_train.shape[1]}")
    print(f"  Output targets   : {y_train.shape[1]}")

    # --- Train all three models ---
    print("\nStep 4: Training models...")
    dt_model = train_decision_tree(x_train, y_train)
    rf_model = train_random_forest(x_train, y_train)
    et_model = train_extra_trees(x_train, y_train)

    # --- Compare R² scores on test set ---
    print("\nStep 5: Comparing R² scores on test set...")
    dt_score = dt_model.score(x_test, y_test)
    rf_score = rf_model.score(x_test, y_test)
    et_score = et_model.score(x_test, y_test)

    print(f"  Decision Tree R² : {dt_score:.4f}")
    print(f"  Random Forest R² : {rf_score:.4f}")
    print(f"  Extra Trees   R² : {et_score:.4f}")

    # --- Save the best model (Extra Trees) and the normalizer ---
    print("\nStep 6: Saving the best model (Extra Trees)...")
    save_model(et_model, os.path.join(model_dir, "extra_trees_model.pkl"))
    save_model(dt_model, os.path.join(model_dir, "decision_tree_model.pkl"))
    save_model(rf_model, os.path.join(model_dir, "random_forest_model.pkl"))

    # Save normalizer — needed for inference to reverse normalization
    normalizer_path = os.path.join(model_dir, "normalizer.csv")
    normalizer_df.to_csv(normalizer_path)
    print(f"  Normalizer saved to: {normalizer_path}")

    print("\nTraining complete!")
