# DataBias Detector

A lightweight MVP to upload a CSV dataset, auto-detect sensitive attributes, compute fairness metrics (Demographic Parity Difference, Equal Opportunity Difference), and visualize results.

## Tech Stack
- Backend: FastAPI, Pandas, Fairlearn, NumPy
- Frontend: Streamlit, Plotly, Requests

## Project Structure
```
databias_detector/
├── backend/
│   ├── main.py              # FastAPI backend (/, /upload, /analyze)
│   ├── bias_analysis.py     # Bias detection and metrics
│   ├── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit dashboard
├── data/
│   └── sample.csv           # Test dataset
└── README.md
```

## Quick Start

### 1) Create venv and install deps (Windows)
```powershell
python -m venv venv
./venv/Scripts/python -m pip install -r backend/requirements.txt
```

### 2) Run Backend
```powershell
./venv/Scripts/python -m uvicorn backend.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for Swagger UI.

### 3) Run Frontend
```powershell
./venv/Scripts/streamlit run frontend/app.py
```
Visit the shown local URL (usually `http://localhost:8501`).

## Usage
- Upload a CSV (e.g., `data/sample.csv`).
- Select a sensitive feature and a binary target column.
- Click "Analyze Bias" to see fairness score, metric breakdown, and charts.

## Notes
- Equal Opportunity Difference requires binary `y_true` and `y_pred`. If predictions are not available, the app falls back to using `y_true` as a proxy.
- For non-binary targets, some metrics may be unavailable.
- Extend with additional metrics, PDF export, and persistence as needed.