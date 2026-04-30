Spacecraft Telemetry Anomaly Detection Project

Overview
This project implements a machine learning pipeline for anomaly detection on the NASA SMAP and MSL spacecraft telemetry datasets.

The workflow:
1. Load telemetry metadata and channel files
2. Preprocess each channel independently
3. Split sequences into sliding windows
4. Label windows using anomaly interval overlap
5. Extract statistical features
6. Train and evaluate Random Forest, XGBoost, and Isolation Forest models

Project Contents
- src/          source code
- notebooks/    experiment notebooks
- data/         dataset files
- README.txt    run instructions
- requirements.txt

Required Data
The project expects the NASA benchmark files:
- labeled_anomalies.csv
- training .npy telemetry files
- test .npy telemetry files

Setup
1. Create and activate a Python environment
2. Install dependencies:

   pip install -r requirements.txt

How to Run
Open the project folder in VS Code or Jupyter Notebook.

Run the notebooks from top to bottom.

Suggested order:
1. Main notebook for strict channel-level experiments
2. SMOTE experiment notebook
3. Optimistic random window-level notebook

Notes
- The notebooks add the project root to sys.path so imports from src/ work correctly.
- Results may vary if channel splits are regenerated differently.
- The SMOTE notebook is exploratory and separate from the main workflow.

Outputs
The project produces:
- processed feature tables
- threshold tuning summaries
- model comparison tables
- evaluation figures

Troubleshooting
- If you get "No module named 'src'", make sure you are running the notebook from the notebooks/ folder inside the project.
- If files are missing, check that the data folder contains the benchmark dataset in the expected structure. 