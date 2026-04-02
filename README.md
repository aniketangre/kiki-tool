# KIKI Lattice Structure Optimizer

ML tool that predicts the optimal lattice structure orientation and cell type for patient-specific biomechanical implants (prosthetics). Given patient and material parameters, it outputs 3D rotation angles and a recommended lattice cell type.

Part of the **KIKI project**, WP 320 — Dissemination.

---

## Quick Start

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 2. Start the server
uvicorn src.app.main:app --reload
```

Then open:

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000/ui` | Interactive browser form (Gradio) |
| `http://127.0.0.1:8000/docs` | REST API documentation (Swagger) |
| `http://127.0.0.1:8000/` | Health check / status |

---

## API Usage

**Endpoint:** `POST /predict`

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "body_wt": 70.0,
    "E1": 15.0, "E2": 15.0, "E3": 15.0,
    "G12": 5.0, "G23": 5.0, "G31": 5.0,
    "NU12": 0.3, "NU23": 0.3, "NU31": 0.3,
    "max_disp": 2.0,
    "max_stress": 150.0,
    "vol_frac": 0.3,
    "vox_to_surf": 0.5
  }'
```

**Response:**

```json
{
  "x_rotation_deg": 12.3456,
  "y_rotation_deg": 45.6789,
  "z_rotation_deg": 90.1234,
  "recommended_cell_type": "GYR"
}
```

---

## Inputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| `body_wt` | kg | Patient body weight |
| `E1`, `E2`, `E3` | GPa | Young's moduli along x, y, z axes |
| `G12`, `G23`, `G31` | GPa | Shear moduli in x-y, y-z, z-x planes |
| `NU12`, `NU23`, `NU31` | — | Poisson's ratios (dimensionless) |
| `max_disp` | mm | Maximum allowable displacement under load |
| `max_stress` | MPa | Maximum allowable internal stress |
| `vol_frac` | — | Volume fraction (0–1) |
| `vox_to_surf` | mm | Distance from voxel grid to part surface |

Material inputs follow **orthotropic material theory** (Voigt notation). Poisson's ratios are validated against physics-based stability constraints (Lempriere, 1968). The engineering constants are converted internally to stiffness tensor entries before inference.

## Outputs

| Field | Description |
|-------|-------------|
| `x_rotation_deg` | Optimal lattice orientation angle around x-axis |
| `y_rotation_deg` | Optimal lattice orientation angle around y-axis |
| `z_rotation_deg` | Optimal lattice orientation angle around z-axis |
| `recommended_cell_type` | Best-suited lattice cell type (e.g. BCC, GYR, FCC, DIA, …) |

Supported cell types: BCC, DTPMS, FCC, FLU, GYR, KEV, OCT, SC, SCH, SPP, DIA, HCG, TEG3, PSM2, RDO, DDK2

---

## Model

- **Algorithm:** Extra Trees Regressor (scikit-learn), selected as best performer over Decision Tree and Random Forest
- **Training data:** FEA simulation results (`data/raw/database_upd.xlsx`)
- **Inputs to model:** 14 features (stiffness tensor entries + structural constraints)
- **Outputs:** 19 values — 3 rotation angles + 16 cell-type probabilities (argmax → recommended cell type)
- **Artifacts:** `src/serving/model/extra_trees_model.pkl` (trained model), `src/serving/model/normalizer.csv` (min/max scaling parameters)

**Retrain the model:**

```bash
python src/models/train.py
```

**Test inference directly:**

```bash
python src/serving/inference.py
```

---

## Project Structure

```
kiki-tool/
├── data/raw/database_upd.xlsx         # Source dataset (FEA simulation results)
├── src/
│   ├── app/main.py                    # FastAPI + Gradio UI (entry point)
│   ├── data/                          # load_data.py, preprocess.py
│   ├── features/build_features.py     # Input/output column definitions
│   ├── models/                        # train.py, evaluate.py, tune.py
│   ├── serving/
│   │   ├── inference.py               # Load model, normalize inputs, predict
│   │   └── model/
│   │       ├── extra_trees_model.pkl  # Trained model
│   │       └── normalizer.csv         # Min/max scaling parameters
│   └── utils/
│       ├── bounds.py                  # Single source of truth for all input bounds
│       └── utils.py                   # Normalization, data splits, material conversion
└── notebooks/model_notebook.ipynb     # Initial model exploration
```

---

## Dependencies

Install all dependencies into the virtual environment:

```bash
pip install -r requirements.txt
```

Key libraries: `scikit-learn`, `fastapi`, `uvicorn`, `gradio`, `pydantic`, `numpy`, `pandas`
