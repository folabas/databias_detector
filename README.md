# DataBias Detector

A lightweight MVP to upload a CSV dataset, auto-detect sensitive attributes, compute fairness metrics (Demographic Parity Difference, Equal Opportunity Difference), and visualize results.

## Built With
- Language: Python 3.x
- Backend: FastAPI, Uvicorn, Pandas, NumPy, Fairlearn
- Frontend: Streamlit, Plotly
- Client: Requests
- Testing: PyTest
- Config: python-dotenv

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

## Detailed Setup (Windows)

### Create and activate virtual environment
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies
```powershell
pip install -r backend\requirements.txt
```

## Environment Variables (.env at project root)
Create a `.env` file in the project root to configure backend behavior.
```env
# Backend service and thresholds
BACKEND_URL=http://127.0.0.1:8000
SPD_THRESHOLD=0.1
DIR_MIN_RATIO=0.8
PED_THRESHOLD=0.05
GROUP_RATIO_MIN=0.5
TARGET_RATIO_MIN=0.6
ALWAYS_ON_TIPS=true

# Optional services
HUGGINGFACE_API_TOKEN=<your_token_if_any>
OLLAMA_BASE_URL=http://127.0.0.1:11434
```
These map to `backend/core/config.py` and are read on startup.

## Run Backend (FastAPI)
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Check API docs at `http://127.0.0.1:8000/docs`.

## Run Frontend (Streamlit)
```powershell
streamlit run frontend\app.py --server.port 8502
```
Visit `http://localhost:8504/`.

## Upload Limits
Streamlit upload limits are set in `.streamlit/config.toml`:
```toml
server.maxUploadSize = 1024
server.maxMessageSize = 1024
```

## Tests
```powershell
python -m pytest backend\tests -q
```

## Troubleshooting
- Backend unreachable: ensure the backend is running at `http://127.0.0.1:8000` and the frontend `BACKEND_URL` points to it.
- Streamlit not updating: stop and rerun the command; clear browser cache if needed.
- Virtual environment not active: the PowerShell prompt should show `(.venv)`; if not, run `.\.venv\Scripts\Activate.ps1`.
- Firewall prompts: allow Python to communicate on local ports.