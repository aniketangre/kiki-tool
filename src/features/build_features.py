# features/build_features.py
# Defines which columns are inputs (features) and which are outputs (targets).


def get_columns(all_params=False):
    """
    Return the column layout used during training and inference.

    The dataset has two types of outputs:
      - Rotation angles: x_rot, y_rot, z_rot (continuous values)
      - Cell type: one-hot encoded columns like BCC, FCC, GYR, etc. (binary)

    Parameters:
        all_params (bool):
            False → use 14 inputs (standard set, recommended)
            True  → use 16 inputs (includes area_to_vol and support_vol)

    Returns:
        reordered_columns (list): All columns to keep from the dataframe.
        input_columns (list): Columns used as model inputs.
        output_columns (list): Columns used as model outputs (targets).
    """
    if not all_params:
        # Standard parameter set: 14 inputs, 19 outputs
        reordered_columns = [
            # --- Inputs (14 columns) ---
            "body_wt (kg)",
            "mat_11 (MPa)", "mat_12 (MPa)", "mat_13 (MPa)",
            "mat_22 (MPa)", "mat_23 (MPa)", "mat_33 (MPa)",
            "mat_44 (MPa)", "mat_55 (MPa)", "mat_66 (MPa)",
            "max_disp (mm)", "max_stress (MPa)",
            "vol_frac", "vox_to_surf (mm)",
            # --- Outputs (19 columns) ---
            "x_rot (deg)", "y_rot (deg)", "z_rot (deg)",
            "BCC", "DTPMS", "FCC", "FLU", "GYR", "KEV", "OCT",
            "SC", "SCH", "SPP", "DIA", "HCG", "TEG3", "PSM2", "RDO", "DDK2",
        ]
        input_columns  = reordered_columns[:14]
        output_columns = reordered_columns[14:]

    else:
        # Full parameter set: 16 inputs, 19 outputs
        reordered_columns = [
            # --- Inputs (16 columns) ---
            "body_wt (kg)",
            "mat_11 (MPa)", "mat_12 (MPa)", "mat_13 (MPa)",
            "mat_22 (MPa)", "mat_23 (MPa)", "mat_33 (MPa)",
            "mat_44 (MPa)", "mat_55 (MPa)", "mat_66 (MPa)",
            "max_disp (mm)", "max_stress (MPa)",
            "vol_frac", "area_to_vol (mm^-1)", "support_vol (mm^3)", "vox_to_surf (mm)",
            # --- Outputs (19 columns) ---
            "x_rot (deg)", "y_rot (deg)", "z_rot (deg)",
            "BCC", "DTPMS", "FCC", "FLU", "GYR", "KEV", "OCT",
            "SC", "SCH", "SPP", "DIA", "HCG", "TEG3", "PSM2", "RDO", "DDK2",
        ]
        input_columns  = reordered_columns[:16]
        output_columns = reordered_columns[16:]

    return reordered_columns, input_columns, output_columns
