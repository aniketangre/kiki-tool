# utils/utils.py
# Shared utility functions used across multiple scripts.

import numpy as np
import pandas as pd


def normalize(df, excluded_columns=None):
    """
    Normalize all numeric columns in a dataframe using Min-Max scaling.
    This scales every value to the range [0, 1].

    Parameters:
        df (pd.DataFrame): Input dataframe to normalize.
        excluded_columns (list): Columns to skip during normalization (e.g. one-hot encoded columns).

    Returns:
        normalized_df (pd.DataFrame): Normalized dataframe.
        normalizer_df (pd.DataFrame): DataFrame with 'max' and 'min' for each column.
                                      Needed later to reverse the normalization.
    """
    # Shuffle the dataframe so data is randomly ordered before splitting
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # Separate the columns we want to normalize from those we want to skip
    if excluded_columns is not None:
        df_numeric = df[df.columns.difference(excluded_columns)]
    else:
        df_numeric = df

    # Apply Min-Max normalization: (x - min) / (max - min)
    normalized_df = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

    # Re-attach the excluded columns as-is (no normalization)
    if excluded_columns is not None:
        for col in excluded_columns:
            normalized_df[col] = df[col]

    # Store the max and min values so we can reverse the normalization later
    normalizer = np.vstack((df_numeric.max(), df_numeric.min())).transpose()
    normalizer_df = pd.DataFrame(normalizer, columns=["max", "min"])
    normalizer_df.index = df_numeric.columns

    return normalized_df, normalizer_df


def get_dataset_partitions(df, train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Split a dataframe into training, validation, and test sets.

    Parameters:
        df (pd.DataFrame): Full normalized dataset.
        train_split (float): Fraction used for training.
        val_split (float): Fraction used for validation.
        test_split (float): Fraction used for testing.

    Returns:
        train_df, val_df, test_df: Three separate dataframes.
    """
    assert (train_split + val_split + test_split) == 1, "Splits must add up to 1.0"

    if not isinstance(df, pd.DataFrame):
        raise Exception("Only pandas DataFrames are accepted!")

    # Calculate the row indices where the splits happen
    n = len(df)
    split1 = int(train_split * n)
    split2 = int((train_split + val_split) * n)

    train_df = df.loc[:split1 - 1]
    val_df   = df.loc[split1:split2 - 1]
    test_df  = df.loc[split2:]

    return train_df, val_df, test_df


def material_props_to_stiffness_constants(E1, E2, E3, G12, G23, G31, NU12, NU23, NU31):
    """
    Convert engineering material properties of an orthotropic material into
    the independent stiffness constants (mat_11 ... mat_66) used as model inputs.

    For an orthotropic material the full 6x6 stiffness tensor C is the inverse
    of the compliance matrix S.  Only the 9 independent components that appear
    on and above the diagonal of the upper-left 3x3 block plus the three shear
    diagonal entries are extracted, because the off-diagonal shear terms are
    zero for a material aligned with its principal axes.

    The stiffness constants map to nTopology's Voigt notation:
        sxx = D1*exx + D2*eyy + D4*ezz
        syy =         D3*eyy + D5*ezz
        szz =                  D6*ezz
        sxy = D10*gxy  (pure shear, D7=D8=D9=0)
        syz = D15*gyz  (pure shear)
        szx = D21*gzx  (pure shear)

    Parameters:
        E1   (float): Young's modulus in the 1-direction (GPa).
        E2   (float): Young's modulus in the 2-direction (GPa).
        E3   (float): Young's modulus in the 3-direction (GPa).
        G12  (float): Shear modulus in the 1-2 plane (GPa).
        G23  (float): Shear modulus in the 2-3 plane (GPa).
        G31  (float): Shear modulus in the 3-1 plane (GPa).
        NU12 (float): Poisson's ratio (contraction in 2 due to extension in 1).
        NU23 (float): Poisson's ratio (contraction in 3 due to extension in 2).
        NU31 (float): Poisson's ratio (contraction in 1 due to extension in 3).

    Returns:
        list[float]: [mat_11, mat_12, mat_13, mat_22, mat_23, mat_33,
                      mat_44, mat_55, mat_66] in MPa.
    """
    # Validation of physical constraints is handled by the Pydantic model
    # in src/app/main.py (MaterialInput.validate_poisson_ratios).
    # This function assumes valid inputs and performs only the math.

    # Convert GPa inputs to Pa for consistent SI calculations
    E1_Pa  = E1  * 1e9
    E2_Pa  = E2  * 1e9
    E3_Pa  = E3  * 1e9
    G12_Pa = G12 * 1e9
    G23_Pa = G23 * 1e9
    G31_Pa = G31 * 1e9

    # Build the 6x6 compliance matrix S (Voigt notation, order: 11,22,33,23,31,12)
    # Normal-stress compliance terms
    S11 =  1.0 / E1_Pa
    S12 = -NU12 / E1_Pa   # symmetry: S21 = S12
    S13 = -NU31 / E1_Pa   # symmetry: S31 = S13
    S22 =  1.0 / E2_Pa
    S23 = -NU23 / E2_Pa   # symmetry: S32 = S23
    S33 =  1.0 / E3_Pa

    # Shear compliance terms (engineering shear strains gij = 2*eij)
    S44 = 1.0 / G23_Pa
    S55 = 1.0 / G31_Pa
    S66 = 1.0 / G12_Pa

    S = np.array([
        [S11, S12, S13,   0,   0,   0],
        [S12, S22, S23,   0,   0,   0],
        [S13, S23, S33,   0,   0,   0],
        [  0,   0,   0, S44,   0,   0],
        [  0,   0,   0,   0, S55,   0],
        [  0,   0,   0,   0,   0, S66],
    ])

    # Invert compliance to get the stiffness tensor C = S^-1
    C = np.linalg.inv(S)

    # Extract the 9 independent stiffness constants and convert Pa -> MPa
    mat_11 = C[0, 0] / 1e6
    mat_12 = C[0, 1] / 1e6
    mat_13 = C[0, 2] / 1e6
    mat_22 = C[1, 1] / 1e6
    mat_23 = C[1, 2] / 1e6
    mat_33 = C[2, 2] / 1e6
    mat_44 = C[3, 3] / 1e6  # shear 2-3
    mat_55 = C[4, 4] / 1e6  # shear 3-1
    mat_66 = C[5, 5] / 1e6  # shear 1-2

    return [mat_11, mat_12, mat_13, mat_22, mat_23, mat_33, mat_44, mat_55, mat_66]


def denormalize(y, normalizer):
    """
    Reverse the Min-Max normalization to get values back in their original scale.

    Parameters:
        y (np.ndarray): Normalized values.
        normalizer (np.ndarray): Array of shape (n_cols, 2) with columns [max, min].

    Returns:
        np.ndarray: Values in original scale.
    """
    return y * (normalizer[:, 0] - normalizer[:, 1]) + normalizer[:, 1]


def compute_data_bounds(db_filepath):
    """
    Compute min/max from the raw database for the five scalar input fields.

    Parameters:
        db_filepath (str): Path to the raw Excel file.

    Returns:
        dict: {field_name: (min, max)} with rounded float values.
    """
    cols = {
        "body_wt":    "body_wt (kg)",
        "max_disp":   "max_disp (mm)",
        "max_stress": "max_stress (MPa)",
        "vol_frac":   "vol_frac",
        "vox_to_surf":"vox_to_surf (mm)",
    }
    df = pd.read_excel(db_filepath, engine="openpyxl").dropna()
    return {
        key: (float(df[col].min()), float(df[col].max()))
        for key, col in cols.items()
    }
