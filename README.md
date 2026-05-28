# Nova-MLW

Course project repository for AI3023 Machine Learning Workshop.

## Project

Kaggle competition: Spaceship Titanic

The final demo pipeline trains an ensemble of XGBoost, LightGBM, and CatBoost models, then writes a Kaggle submission CSV.

## Repository Structure

- `src/spaceship_demo.py` - runnable final demo script.
- `src/spaceship-titanic-project-model_compare.ipynb` - experiment notebook (EDA, baselines, model comparison). EDA charts and results are embedded as cell outputs.
- `reports/Nova@MLW_final_report.docx` - final project report.
- `demo_charts/` - placeholder for any standalone chart exports (charts are primarily in the notebook).
- `data/` - placeholder only; raw data files are not committed.
- `SUBMISSION_CHECKLIST.md` - course requirement checklist.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Data

Download the Spaceship Titanic data from Kaggle:

https://www.kaggle.com/competitions/spaceship-titanic/data

Place these files in **`src/`** (same folder as the demo script):

- `src/train.csv`
- `src/test.csv`

## Run

```bash
python src/spaceship_demo.py
```

Expected outputs (written to `src/`):

- `demo_final_submission.csv` - Kaggle submission file
- `demo_final_parameters.csv` - ensemble parameters log
