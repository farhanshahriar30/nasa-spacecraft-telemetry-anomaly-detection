Spacecraft Telemetry Anomaly Detection Project

1. Overview
This project implements a machine learning pipeline for anomaly detection on the NASA SMAP and MSL spacecraft telemetry datasets.

The workflow:
- load telemetry metadata and raw channel files
- preprocess each channel independently
- split sequences into sliding windows
- assign anomaly labels using interval overlap
- extract statistical features
- train and evaluate anomaly detection models

Models used:
- Random Forest
- XGBoost
- Isolation Forest

2. Project Structure
Typical contents:
- src/               source code modules
- notebooks/         experiment notebooks
- data/              benchmark dataset files
- results/           saved outputs, tables, and figures
- README.txt         run instructions
- requirements.txt   required Python packages

3. Required Data
The project expects the NASA benchmark files:
- labeled_anomalies.csv
- training .npy telemetry files
- test .npy telemetry files

Each telemetry file should be a 2D NumPy array where:
- rows = time steps
- columns = telemetry variables

4. Setup
1. Create and activate a Python environment
2. Install dependencies:

   pip install -r requirements.txt

3. Make sure the data files are placed in the expected project folders

5. How to Run
Open the project folder in VS Code or Jupyter Notebook.

Run the notebooks from top to bottom.

Suggested order:
1. Main notebook for strict channel-level experiments
2. SMOTE experiment notebook
3. Optimistic random window-level notebook

6. Main Workflow
The project pipeline is:
1. Load metadata and telemetry files
2. Preprocess each telemetry channel
3. Create sliding windows
4. Label windows using anomaly interval overlap
5. Extract statistical features
6. Assemble model-ready datasets
7. Train and evaluate models
8. Tune thresholds
9. Generate comparison tables and figures

7. Notes
- The notebooks add the project root to sys.path so imports from src/ work correctly.
- Results may vary if channel splits are regenerated differently.
- The SMOTE notebook contains exploratory experiments only.
- The optimistic notebook uses a random window-level split for comparison with the stricter channel-level setup.

8. Outputs
The project produces:
- processed feature tables
- threshold tuning summaries
- model comparison tables
- evaluation figures

9. Troubleshooting
- If you see "No module named 'src'", make sure you are running the notebook from the notebooks/ folder inside the project.
- If files are missing, confirm that the benchmark dataset exists in the expected data folders.
- If results differ between notebooks, check whether different split settings or thresholds were used.