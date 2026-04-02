# app/main.py
# FastAPI application for the KIKI lattice structure optimizer.
#
# This file exposes the model in two ways:
#   1. REST API  — POST /predict  (for programmatic access)
#   2. Browser UI — GET  /ui      (Gradio form for interactive use)
#
# To run:
#   uvicorn scripts.app.main:app --reload
#
# Then open:
#   http://127.0.0.1:8000/docs   — interactive REST API docs
#   http://127.0.0.1:8000/ui     — user-friendly input form (Gradio)

import math
import os
import sys

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

# Allow imports from sibling folders (serving, utils)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.inference import predict as run_predict, load_model, load_normalizer
from utils.bounds import BOUNDS  # <-- single source of truth for all min/max/default values
from utils.utils import material_props_to_stiffness_constants


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="KIKI Lattice Structure Optimizer",
    description=(
        "Predicts the optimal lattice structure orientation (x, y, z rotation) "
        "and recommended cell type for a given set of biomechanical input parameters. "
        "Based on an Extra Trees regression model trained on FEA simulation data.\n\n"
        "**New in v2:** Inputs are now engineering material constants "
        "(Young's moduli, shear moduli, Poisson's ratios) instead of raw "
        "stiffness tensor entries. The conversion is handled automatically."
    ),
    version="2.0.0",
)

# Load model and normalizer once when the server starts (not on every request).
print("Loading model and normalizer...")
try:
    _model = load_model()
    _normalizer_df = load_normalizer()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    _model = None
    _normalizer_df = None


# ---------------------------------------------------------------------------
# Input schema
#
# Each field uses ge/le constraints pulled directly from BOUNDS so that the
# REST API and the Gradio UI always enforce exactly the same limits.
#
# ge = "greater than or equal to"  → enforces the minimum
# le = "less than or equal to"     → enforces the maximum
# ---------------------------------------------------------------------------
class MaterialInput(BaseModel):
    """
    All input parameters needed to make a prediction.

    Material stiffness is described using standard engineering constants for an
    **orthotropic material** — a material that has different properties along
    each of its three principal axes (x, y, z).

    These are converted internally to the stiffness tensor the model expects.
    """

    # --- Patient / loading ---
    body_wt: float = Field(
        default=BOUNDS.body_wt.default,
        ge=BOUNDS.body_wt.min,
        le=BOUNDS.body_wt.max,
        description=(
            f"Body weight of the patient (kg). "
            f"Allowed range: {BOUNDS.body_wt.min}–{BOUNDS.body_wt.max} kg."
        ),
        json_schema_extra={"example": BOUNDS.body_wt.default},
    )

    # --- Young's moduli: how stiff the material is along each axis ---
    E1: float = Field(
        default=BOUNDS.E1.default,
        ge=BOUNDS.E1.min,
        le=BOUNDS.E1.max,
        description=(
            f"Young's modulus along the x-axis (GPa). "
            f"Describes stiffness when pulled/compressed along x. "
            f"Allowed range: {BOUNDS.E1.min}–{BOUNDS.E1.max} GPa."
        ),
        json_schema_extra={"example": BOUNDS.E1.default},
    )
    E2: float = Field(
        default=BOUNDS.E2.default,
        ge=BOUNDS.E2.min,
        le=BOUNDS.E2.max,
        description=(
            f"Young's modulus along the y-axis (GPa). "
            f"Allowed range: {BOUNDS.E2.min}–{BOUNDS.E2.max} GPa."
        ),
        json_schema_extra={"example": BOUNDS.E2.default},
    )
    E3: float = Field(
        default=BOUNDS.E3.default,
        ge=BOUNDS.E3.min,
        le=BOUNDS.E3.max,
        description=(
            f"Young's modulus along the z-axis (GPa). "
            f"Allowed range: {BOUNDS.E3.min}–{BOUNDS.E3.max} GPa."
        ),
        json_schema_extra={"example": BOUNDS.E3.default},
    )

    # --- Shear moduli: resistance to shear in each plane ---
    G12: float = Field(
        default=BOUNDS.G12.default,
        ge=BOUNDS.G12.min,
        le=BOUNDS.G12.max,
        description=(
            f"Shear modulus in the x-y plane (GPa). "
            f"Allowed range: {BOUNDS.G12.min}–{BOUNDS.G12.max} GPa."
        ),
        json_schema_extra={"example": BOUNDS.G12.default},
    )
    G23: float = Field(
        default=BOUNDS.G23.default,
        ge=BOUNDS.G23.min,
        le=BOUNDS.G23.max,
        description=(
            f"Shear modulus in the y-z plane (GPa). "
            f"Allowed range: {BOUNDS.G23.min}–{BOUNDS.G23.max} GPa."
        ),
        json_schema_extra={"example": BOUNDS.G23.default},
    )
    G31: float = Field(
        default=BOUNDS.G31.default,
        ge=BOUNDS.G31.min,
        le=BOUNDS.G31.max,
        description=(
            f"Shear modulus in the z-x plane (GPa). "
            f"Allowed range: {BOUNDS.G31.min}–{BOUNDS.G31.max} GPa."
        ),
        json_schema_extra={"example": BOUNDS.G31.default},
    )

    # --- Poisson's ratios: how much the material contracts sideways when stretched ---
    # These have additional physics-based constraints checked in validate_poisson_ratios below.
    NU12: float = Field(
        default=BOUNDS.NU12.default,
        ge=BOUNDS.NU12.min,
        le=BOUNDS.NU12.max,
        description=(
            f"Poisson's ratio: contraction in y when stretched in x (dimensionless). "
            f"Allowed range: {BOUNDS.NU12.min}–{BOUNDS.NU12.max}. "
            "Additional physics constraints also apply depending on E1 and E2."
        ),
        json_schema_extra={"example": BOUNDS.NU12.default},
    )
    NU23: float = Field(
        default=BOUNDS.NU23.default,
        ge=BOUNDS.NU23.min,
        le=BOUNDS.NU23.max,
        description=(
            f"Poisson's ratio: contraction in z when stretched in y (dimensionless). "
            f"Allowed range: {BOUNDS.NU23.min}–{BOUNDS.NU23.max}. "
            "Additional physics constraints also apply depending on E2 and E3."
        ),
        json_schema_extra={"example": BOUNDS.NU23.default},
    )
    NU31: float = Field(
        default=BOUNDS.NU31.default,
        ge=BOUNDS.NU31.min,
        le=BOUNDS.NU31.max,
        description=(
            f"Poisson's ratio: contraction in x when stretched in z (dimensionless). "
            f"Allowed range: {BOUNDS.NU31.min}–{BOUNDS.NU31.max}. "
            "Additional physics constraints also apply depending on E3 and E1."
        ),
        json_schema_extra={"example": BOUNDS.NU31.default},
    )

    # --- Structural design constraints ---
    max_disp: float = Field(
        default=BOUNDS.max_disp.default,
        ge=BOUNDS.max_disp.min,
        le=BOUNDS.max_disp.max,
        description=(
            f"Maximum allowable displacement under load (mm). "
            f"Allowed range: {BOUNDS.max_disp.min}–{BOUNDS.max_disp.max} mm."
        ),
        json_schema_extra={"example": BOUNDS.max_disp.default},
    )
    max_stress: float = Field(
        default=BOUNDS.max_stress.default,
        ge=BOUNDS.max_stress.min,
        le=BOUNDS.max_stress.max,
        description=(
            f"Maximum allowable internal stress (MPa). "
            f"Allowed range: {BOUNDS.max_stress.min}–{BOUNDS.max_stress.max} MPa."
        ),
        json_schema_extra={"example": BOUNDS.max_stress.default},
    )
    vol_frac: float = Field(
        default=BOUNDS.vol_frac.default,
        ge=BOUNDS.vol_frac.min,
        le=BOUNDS.vol_frac.max,
        description=(
            f"Volume fraction: ratio of solid material to total bounding volume "
            f"(dimensionless). Allowed range: {BOUNDS.vol_frac.min}–{BOUNDS.vol_frac.max}."
        ),
        json_schema_extra={"example": BOUNDS.vol_frac.default},
    )
    vox_to_surf: float = Field(
        default=BOUNDS.vox_to_surf.default,
        ge=BOUNDS.vox_to_surf.min,
        le=BOUNDS.vox_to_surf.max,
        description=(
            f"Distance from the voxel grid to the part surface (mm). "
            f"Allowed range: {BOUNDS.vox_to_surf.min}–{BOUNDS.vox_to_surf.max} mm."
        ),
        json_schema_extra={"example": BOUNDS.vox_to_surf.default},
    )

    # -----------------------------------------------------------------------
    # Physics-based validation for Poisson's ratios
    #
    # For an orthotropic material the compliance matrix S must be positive
    # definite — meaning the material always stores (not creates) energy.
    # This imposes two constraints on the Poisson's ratios:
    #
    #   1. Pairwise bound  : |NUij| < sqrt(Ei / Ej)
    #      (from each 2×2 sub-block of S having a positive determinant)
    #
    #   2. Global condition: 1 - NU12·NU21 - NU23·NU32 - NU31·NU13
    #                          - 2·NU12·NU23·NU31 > 0
    #      (from the full 3×3 normal-stress block of S being positive definite)
    #
    # These constraints cannot be stored as simple min/max bounds because they
    # depend on the combination of E values provided at runtime.
    #
    # Source: Lempriere, B.M. (1968). "Poisson's ratio in orthotropic materials."
    #         AIAA Journal, 6(11), 2226–2227. https://doi.org/10.2514/3.4974
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_poisson_ratios(self):
        E1, E2, E3 = self.E1, self.E2, self.E3
        NU12, NU23, NU31 = self.NU12, self.NU23, self.NU31

        # --- Constraint 1: Pairwise bounds ---
        # Each pair (NUij, Ei, Ej) must satisfy |NUij| < sqrt(Ei/Ej).
        pairwise = [
            ("NU12", NU12, E1, E2, "E1", "E2"),
            ("NU23", NU23, E2, E3, "E2", "E3"),
            ("NU31", NU31, E3, E1, "E3", "E1"),
        ]
        for name, nu, Ei, Ej, Ei_name, Ej_name in pairwise:
            # Avoid division-by-zero if a modulus is exactly 0
            if Ej == 0:
                continue
            bound = math.sqrt(Ei / Ej)
            if abs(nu) >= bound:
                raise ValueError(
                    f"{name} = {nu:.4f} is out of the physical range. "
                    f"For these moduli, |{name}| must be < sqrt({Ei_name}/{Ej_name}) = {bound:.4f}. "
                    f"Tip: reduce |{name}| or adjust {Ei_name} / {Ej_name}."
                )

        # --- Constraint 2: Global determinant condition ---
        # The reciprocal Poisson's ratios follow from symmetry of S (NUij/Ei = NUji/Ej).
        NU21 = NU12 * (E2 / E1) if E1 != 0 else 0
        NU32 = NU23 * (E3 / E2) if E2 != 0 else 0
        NU13 = NU31 * (E1 / E3) if E3 != 0 else 0

        det_value = (
            1
            - NU12 * NU21
            - NU23 * NU32
            - NU31 * NU13
            - 2 * NU12 * NU23 * NU31
        )
        if det_value <= 0:
            raise ValueError(
                f"The combination of Poisson's ratios is physically impossible "
                f"(global stability value = {det_value:.6f}, must be > 0). "
                "The three Poisson's ratios together make the material unstable. "
                "Try reducing their magnitudes."
            )

        return self


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class PredictionOutput(BaseModel):
    x_rotation_deg: float
    y_rotation_deg: float
    z_rotation_deg: float
    recommended_cell_type: str


# ---------------------------------------------------------------------------
# Helper: build the 14-value list the model expects from MaterialInput
# ---------------------------------------------------------------------------
def build_model_input(data: MaterialInput) -> list:
    """
    Convert a MaterialInput (engineering constants) into the 14-element list
    that inference.py expects.

    The model was trained on stiffness tensor constants (mat_11 … mat_66, in MPa),
    so we first convert the engineering properties using
    material_props_to_stiffness_constants, then assemble the full input list.
    """
    mat = material_props_to_stiffness_constants(
        data.E1, data.E2, data.E3,
        data.G12, data.G23, data.G31,
        data.NU12, data.NU23, data.NU31,
    )
    # mat = [mat_11, mat_12, mat_13, mat_22, mat_23, mat_33, mat_44, mat_55, mat_66]

    return [
        data.body_wt,     # body_wt    (kg)
        mat[0],           # mat_11     (MPa)
        mat[1],           # mat_12     (MPa)
        mat[2],           # mat_13     (MPa)
        mat[3],           # mat_22     (MPa)
        mat[4],           # mat_23     (MPa)
        mat[5],           # mat_33     (MPa)
        mat[6],           # mat_44     (MPa)
        mat[7],           # mat_55     (MPa)
        mat[8],           # mat_66     (MPa)
        data.max_disp,    # max_disp   (mm)
        data.max_stress,  # max_stress (MPa)
        data.vol_frac,    # vol_frac
        data.vox_to_surf, # vox_to_surf (mm)
    ]


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    """Returns API status and whether the model is loaded."""
    return {
        "status": "KIKI API is running",
        "model_loaded": _model is not None,
        "tip": "Visit /ui for the interactive browser form, or /docs for the API reference.",
    }


@app.post("/predict", response_model=PredictionOutput,
          summary="Predict lattice orientation and cell type")
def predict_endpoint(input_data: MaterialInput):
    """
    Predict the optimal lattice structure orientation and cell type.

    **Input:** Engineering material constants (Young's moduli, shear moduli,
    Poisson's ratios) plus structural design constraints.

    **Output:**
    - `x_rotation_deg`, `y_rotation_deg`, `z_rotation_deg`: Optimal orientation.
    - `recommended_cell_type`: Best-suited lattice cell type (e.g. BCC, GYR, FCC).
    """
    if _model is None or _normalizer_df is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Please train the model first: python src/models/train.py"
            ),
        )
    try:
        result = run_predict(
            build_model_input(input_data),
            model=_model,
            normalizer_df=_normalizer_df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return result


@app.get("/model-info", summary="Get information about the loaded model")
def model_info():
    """Returns metadata about the currently loaded model."""
    if _model is None:
        return {"status": "No model is currently loaded."}
    return {
        "model_type":    type(_model).__name__,
        "n_estimators":  getattr(_model, "n_estimators", "N/A"),
        "max_depth":     getattr(_model, "max_depth", "N/A"),
        "n_outputs":     getattr(_model, "n_outputs_", "N/A"),
        "n_features_in": getattr(_model, "n_features_in_", "N/A"),
    }


# ---------------------------------------------------------------------------
# Gradio UI — mounted at /ui
#
# Each input is a gr.Slider, which:
#   - Reads min / max / default / step directly from BOUNDS
#   - Prevents the user from entering out-of-range values via the UI
#   - Shows the allowed range visually on the slider track
# ---------------------------------------------------------------------------

def gradio_predict(body_wt, E1, E2, E3, G12, G23, G31,
                   NU12, NU23, NU31,
                   max_disp, max_stress, vol_frac, vox_to_surf):
    """
    Called when the user clicks 'Predict' in the browser form.
    Reuses the Pydantic model so all validation logic (including physics
    checks for Poisson's ratios) stays in one place.
    """
    # Run Pydantic validation — same rules as the REST API
    try:
        input_data = MaterialInput(
            body_wt=body_wt,
            E1=E1, E2=E2, E3=E3,
            G12=G12, G23=G23, G31=G31,
            NU12=NU12, NU23=NU23, NU31=NU31,
            max_disp=max_disp, max_stress=max_stress,
            vol_frac=vol_frac, vox_to_surf=vox_to_surf,
        )
    except Exception as e:
        # Show the validation error in the cell-type output box
        return f"Input error: {e}", "", "", ""

    if _model is None or _normalizer_df is None:
        return "Model not loaded. Run: python src/models/train.py", "", "", ""

    try:
        result = run_predict(
            build_model_input(input_data),
            model=_model,
            normalizer_df=_normalizer_df,
        )
    except Exception as e:
        return f"Prediction error: {e}", "", "", ""

    return (
        result["recommended_cell_type"],
        f"{result['x_rotation_deg']}°",
        f"{result['y_rotation_deg']}°",
        f"{result['z_rotation_deg']}°",
    )


# ---------------------------------------------------------------------------
# Helper: build a gr.Slider from a FieldBounds object
#
# Centralises slider creation so the label, min, max, default, and step
# are always taken from the same BOUNDS object that drives Pydantic.
# ---------------------------------------------------------------------------
def make_slider(bounds, label: str, info: str) -> gr.Slider:
    """
    Create a Gradio slider whose range matches the given FieldBounds entry.

    Args:
        bounds : A FieldBounds named-tuple from BOUNDS (e.g. BOUNDS.body_wt).
        label  : Text label shown above the slider in the UI.
        info   : Short explanation shown as a tooltip / sub-label.

    Returns:
        A configured gr.Slider component.
    """
    return gr.Slider(
        minimum=bounds.min,
        maximum=bounds.max,
        value=bounds.default,
        step=bounds.step,
        label=label,
        info=info,
    )


with gr.Blocks(title="KIKI Lattice Optimizer") as gradio_ui:

    gr.Markdown("""
    # KIKI Lattice Structure Optimizer
    Predicts the optimal **lattice cell type** and **orientation angles** for your design.

    Fill in the material properties and structural constraints below, then click **Predict**.
    Each slider shows the allowed range and snaps to the nearest valid step.
    """)

    with gr.Row():

        # --- Left column: patient data + material stiffness ---
        with gr.Column():

            gr.Markdown("### Patient & Loading")
            body_wt_in = make_slider(
                BOUNDS.body_wt,
                label="Body Weight (kg)",
                info=f"Patient body weight. Range: {BOUNDS.body_wt.min}–{BOUNDS.body_wt.max} kg.",
            )

            gr.Markdown("### Young's Moduli (GPa)")
            gr.Markdown(
                "_How stiff the material is along each axis. "
                "Higher value = harder to stretch in that direction._"
            )
            E1_in = make_slider(
                BOUNDS.E1,
                label="E1  —  x-axis",
                info=f"Stiffness along x. Range: {BOUNDS.E1.min}–{BOUNDS.E1.max} GPa.",
            )
            E2_in = make_slider(
                BOUNDS.E2,
                label="E2  —  y-axis",
                info=f"Stiffness along y. Range: {BOUNDS.E2.min}–{BOUNDS.E2.max} GPa.",
            )
            E3_in = make_slider(
                BOUNDS.E3,
                label="E3  —  z-axis",
                info=f"Stiffness along z. Range: {BOUNDS.E3.min}–{BOUNDS.E3.max} GPa.",
            )

            gr.Markdown("### Shear Moduli (GPa)")
            gr.Markdown(
                "_Resistance to sliding deformation in each plane._"
            )
            G12_in = make_slider(
                BOUNDS.G12,
                label="G12  —  x-y plane",
                info=f"Shear stiffness in the x-y plane. Range: {BOUNDS.G12.min}–{BOUNDS.G12.max} GPa.",
            )
            G23_in = make_slider(
                BOUNDS.G23,
                label="G23  —  y-z plane",
                info=f"Shear stiffness in the y-z plane. Range: {BOUNDS.G23.min}–{BOUNDS.G23.max} GPa.",
            )
            G31_in = make_slider(
                BOUNDS.G31,
                label="G31  —  z-x plane",
                info=f"Shear stiffness in the z-x plane. Range: {BOUNDS.G31.min}–{BOUNDS.G31.max} GPa.",
            )

        # --- Right column: Poisson's ratios + structural constraints ---
        with gr.Column():

            gr.Markdown("### Poisson's Ratios (dimensionless)")
            gr.Markdown(
                "_How much the material contracts sideways when stretched along one axis. "
                "The slider enforces simple min/max bounds. An additional physics check "
                "runs on submission and will show an error if the combination of values "
                "is physically impossible._"
            )
            NU12_in = make_slider(
                BOUNDS.NU12,
                label="NU12  —  stretch x → contract y",
                info=f"Range: {BOUNDS.NU12.min}–{BOUNDS.NU12.max}. Physics check also applies.",
            )
            NU23_in = make_slider(
                BOUNDS.NU23,
                label="NU23  —  stretch y → contract z",
                info=f"Range: {BOUNDS.NU23.min}–{BOUNDS.NU23.max}. Physics check also applies.",
            )
            NU31_in = make_slider(
                BOUNDS.NU31,
                label="NU31  —  stretch z → contract x",
                info=f"Range: {BOUNDS.NU31.min}–{BOUNDS.NU31.max}. Physics check also applies.",
            )

            gr.Markdown("### Structural Constraints")
            max_disp_in = make_slider(
                BOUNDS.max_disp,
                label="Max Displacement (mm)",
                info=f"Maximum deformation allowed under load. Range: {BOUNDS.max_disp.min}–{BOUNDS.max_disp.max} mm.",
            )
            max_stress_in = make_slider(
                BOUNDS.max_stress,
                label="Max Stress (MPa)",
                info=f"Maximum internal stress allowed. Range: {BOUNDS.max_stress.min}–{BOUNDS.max_stress.max} MPa.",
            )
            vol_frac_in = make_slider(
                BOUNDS.vol_frac,
                label="Volume Fraction  (0 – 1)",
                info=f"Fraction of solid material in the bounding volume. Range: {BOUNDS.vol_frac.min}–{BOUNDS.vol_frac.max}.",
            )
            vox_to_surf_in = make_slider(
                BOUNDS.vox_to_surf,
                label="Voxel-to-Surface Distance (mm)",
                info=f"Distance from the voxel grid to the part surface. Range: {BOUNDS.vox_to_surf.min}–{BOUNDS.vox_to_surf.max} mm.",
            )

    predict_btn = gr.Button("Predict", variant="primary")

    gr.Markdown("### Prediction Results")
    with gr.Row():
        out_cell  = gr.Textbox(label="Recommended Cell Type")
        out_x_rot = gr.Textbox(label="X Rotation")
        out_y_rot = gr.Textbox(label="Y Rotation")
        out_z_rot = gr.Textbox(label="Z Rotation")

    predict_btn.click(
        fn=gradio_predict,
        inputs=[
            body_wt_in,
            E1_in, E2_in, E3_in,
            G12_in, G23_in, G31_in,
            NU12_in, NU23_in, NU31_in,
            max_disp_in, max_stress_in, vol_frac_in, vox_to_surf_in,
        ],
        outputs=[out_cell, out_x_rot, out_y_rot, out_z_rot],
    )

# Mount the Gradio UI inside the FastAPI app at /ui
app = gr.mount_gradio_app(app, gradio_ui, path="/ui")
