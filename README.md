# Nova-MLW

Course project repository for AI3023 Machine Learning Workshop.

## Project

Kaggle competition: Spaceship Titanic

The final demo pipeline trains an ensemble of XGBoost, LightGBM, and CatBoost models, then writes a Kaggle submission CSV.

## Repository Structure

- `src/spaceship_demo_code.py` - runnable final demo script.
- `notebooks/spaceship-titanic-project.ipynb` - experiment notebook.
- `reports/Spaceship_Titanic_Optimization_Report_REVISED_with_Static_Tuning_Detail.docx` - project report draft/material.
- `data/` - place Kaggle input files here locally. The raw dataset is not committed.
- `SUBMISSION_CHECKLIST.md` - course requirement checklist and missing-material notes.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Data

Download the Spaceship Titanic data from Kaggle:

https://www.kaggle.com/competitions/spaceship-titanic/data

Place these files in `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Run

From the repository root:

```bash
python src/spaceship_demo_code.py --train data/train.csv --test data/test.csv --submission data/sample_submission.csv
```

Expected outputs:

- `demo_final_submission_static_81201.csv`
- `demo_final_parameters.csv`
