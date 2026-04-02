# KIKI Tool — Claude Code Context

## Project Purpose
ML application that predicts optimal lattice structure designs for biomechanical implants (prosthetics). Given patient and material parameters, it recommends 3D lattice orientation angles and a lattice cell type.

## How to Run
```bash
# From project root, activate venv first
.venv\Scripts\activate   # Windows
# Then start the API server
uvicorn src.app.main:app --reload
```
- REST API: `POST /predict`
- Interactive browser UI: `GET /ui` (Gradio)
- API docs: `GET /docs`

## Architecture
```
kiki-tool/
├── data/raw/database_upd.xlsx     # Source dataset (FEA simulation results)
├── src/
│   ├── app/main.py                # FastAPI + Gradio UI (entry point)
│   ├── data/                      # load_data.py, preprocess.py
│   ├── features/build_features.py # Input/output column definitions
│   ├── models/                    # train.py, evaluate.py, tune.py
│   ├── serving/
│   │   ├── inference.py           # Load model, normalize, predict
│   │   └── model/
│   │       ├── extra_trees_model.pkl  # Trained model (437 MB)
│   │       └── normalizer.csv         # Min/max scaling params
│   └── utils/
│       ├── bounds.py              # Single source of truth for all input bounds
│       └── utils.py               # Normalization, splits, material conversion
├── trained_models/model_22v.joblib  # Older pre-trained model (legacy)
└── notebooks/model_notebook.ipynb  # Initial model exploration (not actively used)
```

## Model
- Algorithm: Extra Trees Regressor (best of Decision Tree, Random Forest, Extra Trees)
- Inputs (14): `body_wt`, `E1/E2/E3` (Young's moduli), `G12/G23/G31` (shear moduli), `NU12/NU23/NU31` (Poisson's ratios), `max_disp`, `max_stress`, `vol_frac`, `vox_to_surf`
- Outputs (19): 3 rotation angles (x, y, z in degrees) + 16 cell-type probabilities (BCC, FCC, GYR, DIA, etc.)
- Retrain: run `python src/models/train.py`

## Domain Notes
- Material inputs follow orthotropic material theory (Voigt notation)
- Poisson's ratio constraints enforced per Lempriere (1968)
- `utils.py` converts engineering constants (E, G, ν) → stiffness tensor (mat_11…mat_66)
- `bounds.py` is the single source of truth for all input ranges — change bounds there, not elsewhere

## Key Decisions Made
- Extra Trees selected as production model (best R² on validation set)
- Gradio UI added on top of FastAPI for interactive demos/dissemination
- Material property conversion moved from `material_prop_to_tensor.py` into `utils.py` (legacy file kept for reference)
- Serving model stored in `src/serving/model/` (not `trained_models/`)
