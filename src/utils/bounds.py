# utils/bounds.py
#
# Single source of truth for all input field bounds used by the KIKI optimizer.
#
# HOW TO USE:
#   from utils.bounds import BOUNDS
#
#   BOUNDS.body_wt.min      # minimum allowed value
#   BOUNDS.body_wt.max      # maximum allowed value
#   BOUNDS.body_wt.default  # suggested starting value
#   BOUNDS.body_wt.step     # slider step size (used in the Gradio UI)
#
# WHY A CENTRAL BOUNDS FILE?
#   Keeping all bounds here means you only need to change one place to update
#   validation in both the REST API (Pydantic) and the browser UI (Gradio).
#   No more hunting through main.py to find where a limit was set.

from typing import NamedTuple


# ---------------------------------------------------------------------------
# FieldBounds — stores the range and display hints for a single input field
# ---------------------------------------------------------------------------

class FieldBounds(NamedTuple):
    """
    Describes the allowed range for one numeric input field.

    Attributes:
        min     : Smallest value the field may take (inclusive).
        max     : Largest value the field may take (inclusive).
        default : Sensible starting value shown in the UI and API examples.
        step    : Increment used by the Gradio slider (controls slider precision).
    """
    min:     float
    max:     float
    default: float
    step:    float


# ---------------------------------------------------------------------------
# InputBounds — collects FieldBounds for every input the model expects
# ---------------------------------------------------------------------------

class InputBounds:
    """
    All input bounds for the KIKI Lattice Structure Optimizer.

    Each attribute is a FieldBounds named-tuple for one model input.
    Edit the numbers below to change allowed ranges everywhere at once.
    """

    # -----------------------------------------------------------------------
    # Patient & Loading
    # -----------------------------------------------------------------------
    body_wt = FieldBounds(
        min=55.0, max=95.0, default=75.0, step=1.0
        # Unit: kg  |  Range from database (55–95 kg)
    )

    # -----------------------------------------------------------------------
    # Young's Moduli  (GPa)
    # How stiff the material is along each principal axis.
    # Higher value = harder to stretch in that direction.
    # -----------------------------------------------------------------------
    E1 = FieldBounds(
        min=0.0, max=50.0, default=21.0, step=0.5
        # Stiffness along the x-axis
    )
    E2 = FieldBounds(
        min=0.0, max=50.0, default=10.0, step=0.5
        # Stiffness along the y-axis
    )
    E3 = FieldBounds(
        min=0.0, max=50.0, default=10.0, step=0.5
        # Stiffness along the z-axis
    )

    # -----------------------------------------------------------------------
    # Shear Moduli  (GPa)
    # Resistance to sliding deformation in each material plane.
    # -----------------------------------------------------------------------
    G12 = FieldBounds(
        min=0.0, max=30.0, default=6.0, step=0.5
        # Shear in the x-y plane
    )
    G23 = FieldBounds(
        min=0.0, max=30.0, default=4.0, step=0.5
        # Shear in the y-z plane
    )
    G31 = FieldBounds(
        min=0.0, max=30.0, default=6.0, step=0.5
        # Shear in the z-x plane
    )

    # -----------------------------------------------------------------------
    # Poisson's Ratios  (dimensionless)
    # How much the material contracts sideways when stretched along one axis.
    # NOTE: physics-based pairwise and global constraints are checked separately
    #       in main.py (they depend on the combination of E values at runtime).
    # -----------------------------------------------------------------------
    NU12 = FieldBounds(
        min=0.1, max=0.45, default=0.3, step=0.01
        # Contraction in y when stretched in x
    )
    NU23 = FieldBounds(
        min=0.1, max=0.45, default=0.2, step=0.01
        # Contraction in z when stretched in y
    )
    NU31 = FieldBounds(
        min=0.1, max=0.45, default=0.3, step=0.01
        # Contraction in x when stretched in z
    )

    # -----------------------------------------------------------------------
    # Structural Constraints
    # -----------------------------------------------------------------------
    max_disp = FieldBounds(
        min=0.0, max=25, default=5.0, step=0.5
        # Unit: mm  |  Range from database (0.005–310.21 mm)
    )
    max_stress = FieldBounds(
        min=25, max=250, default=50, step=1.0
        # Unit: MPa  |  Range from database (34.2–6992.87 MPa)
    )
    vol_frac = FieldBounds(
        min=0.20, max=0.48, default=0.25, step=0.01
        # Dimensionless  |  Range from database (0.20–0.48)
    )
    vox_to_surf = FieldBounds(
        min=60.0, max=200.0, default=100.0, step=1.0
        # Unit: mm  |  Range from database (60.33–240.76 mm)
    )

# Singleton — import this object wherever bounds are needed.
# Example:  from utils.bounds import BOUNDS
BOUNDS = InputBounds()
