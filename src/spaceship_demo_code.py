"""
Spaceship Titanic Final Demo Code: Retrain-Only Fixed Static Ensemble
=======================================================

Purpose
-------
This is the final demo version of the model pipeline. It does NOT perform
parameter search. It directly uses the best static ensemble parameters found
in the previous optimization experiments and outputs one Kaggle submission file.

Final fixed ensemble parameters from the best public-feedback experiments:
    XGBoost weight      = 0.0075
    LightGBM weight     = 0.2325
    CatBoost weight     = 0.7600
    probability alpha   = 0.8125
    decision threshold  = 0.5050

The final blend is:
    final_score = alpha * probability_blend + (1 - alpha) * rank_blend

Output:
    demo_final_submission_static_81201.csv
    demo_final_parameters.csv

Run:
    python spaceship_demo_static_final.py

Optional:
    python spaceship_demo_static_final.py --train train.csv --test test.csv --submission sample_submission.csv

Important:
    This version does NOT read or reuse any previous prediction files.
    It always trains XGBoost, LightGBM, and CatBoost from train.csv,
    predicts test.csv, and then applies the fixed static ensemble formula.

Required packages for full reproduction:
    pandas numpy scikit-learn xgboost lightgbm catboost
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    XGB_IMPORT_ERROR = e

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    LGBMClassifier = None
    LGBM_IMPORT_ERROR = e

try:
    from catboost import CatBoostClassifier
except Exception as e:
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = e


RANDOM_STATE = 42

# ==========================
# Final fixed best parameters
# ==========================
W_XGBOOST = 0.0075
W_LIGHTGBM = 0.2325
W_CATBOOST = 0.7600
ALPHA_PROBABILITY = 0.8125
DECISION_THRESHOLD = 0.5050
N_SPLITS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve_path(user_path: str | None, candidate_names: List[str], required: bool = True) -> str | None:
    search_paths: List[Path] = []
    if user_path:
        search_paths.append(Path(user_path))

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for name in candidate_names:
        search_paths.append(script_dir / name)
        search_paths.append(cwd / name)
        search_paths.append(Path("/mnt/data") / name)

    for p in search_paths:
        if p.exists():
            return str(p)

    if required:
        raise FileNotFoundError("Cannot find required file. Tried: " + ", ".join(candidate_names))
    return None


def make_onehot_encoder() -> OneHotEncoder:
    # Compatible with different scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for Spaceship Titanic."""
    df = df.copy()

    if "PassengerId" in df.columns:
        split_id = df["PassengerId"].astype(str).str.split("_", expand=True)
        df["GroupId"] = split_id[0]
        df["PassengerNo"] = pd.to_numeric(split_id[1], errors="coerce")
    else:
        df["GroupId"] = "Missing"
        df["PassengerNo"] = np.nan

    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).replace("nan", np.nan).str.split("/", expand=True)
        df["CabinDeck"] = cabin_split[0]
        df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        df["CabinSide"] = cabin_split[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan

    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype(str).str.split().str[-1]
        df.loc[df["Name"].isna(), "Surname"] = np.nan
        df["NameLength"] = df["Name"].astype(str).str.len()
    else:
        df["Surname"] = np.nan
        df["NameLength"] = np.nan

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["TotalSpend"] = df[spend_cols].sum(axis=1, skipna=True)
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)
    df["LuxurySpend"] = df[["FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1, skipna=True)
    df["BasicSpend"] = df["RoomService"].fillna(0)
    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1)

    for col in spend_cols + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
        df[f"Log_{col}"] = np.log1p(df[col].fillna(0))

    if "Age" in df.columns:
        df["AgeBin"] = pd.cut(
            df["Age"],
            bins=[-1, 12, 18, 30, 45, 60, 120],
            labels=["child", "teen", "young", "adult", "middle", "senior"],
        ).astype("object")
        df["IsChild"] = (df["Age"].fillna(-1) < 13).astype(int)
        df["IsSenior"] = (df["Age"].fillna(-1) >= 60).astype(int)
    else:
        df["AgeBin"] = np.nan
        df["IsChild"] = 0
        df["IsSenior"] = 0

    if "CryoSleep" in df.columns:
        cryo_true = df["CryoSleep"].astype(str).str.lower().eq("true")
        df["CryoAndSpend"] = (cryo_true & (df["TotalSpend"].fillna(0) > 0)).astype(int)
        df["CryoAndNoSpend"] = (cryo_true & (df["TotalSpend"].fillna(0) == 0)).astype(int)
    else:
        df["CryoAndSpend"] = 0
        df["CryoAndNoSpend"] = 0

    return df


def add_count_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add count features using train + test together. This does not use labels."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["__is_train__"] = 1
    test_df["__is_train__"] = 0
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    if "GroupId" in combined.columns:
        combined["GroupSize"] = combined.groupby("GroupId")["PassengerId"].transform("count")
        combined["IsSolo"] = (combined["GroupSize"] == 1).astype(int)
    else:
        combined["GroupSize"] = 1
        combined["IsSolo"] = 1

    if "Surname" in combined.columns:
        combined["SurnameSize"] = combined.groupby("Surname")["PassengerId"].transform("count")
    else:
        combined["SurnameSize"] = 1

    train_new = combined[combined["__is_train__"] == 1].drop(columns=["__is_train__"])
    test_new = combined[combined["__is_train__"] == 0].drop(columns=["__is_train__"])
    return train_new.reset_index(drop=True), test_new.reset_index(drop=True)


def prepare_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "Transported" not in train_df.columns:
        raise ValueError("Training file must contain target column: Transported")

    y = train_df["Transported"].astype(bool).astype(int)
    test_ids = test_df["PassengerId"].copy()

    train_features = train_df.drop(columns=["Transported"])
    train_features = add_features(train_features)
    test_features = add_features(test_df)
    train_features, test_features = add_count_features(train_features, test_features)

    drop_cols = ["PassengerId", "Name", "Cabin"]
    train_features = train_features.drop(columns=[c for c in drop_cols if c in train_features.columns])
    test_features = test_features.drop(columns=[c for c in drop_cols if c in test_features.columns])

    return train_features, y, test_features, test_ids


def get_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Safely split columns into numeric and categorical columns."""
    from pandas.api.types import (
        is_bool_dtype,
        is_categorical_dtype,
        is_numeric_dtype,
        is_object_dtype,
        is_string_dtype,
    )

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in X.columns:
        s = X[col]
        if is_object_dtype(s) or is_string_dtype(s) or is_categorical_dtype(s) or is_bool_dtype(s):
            categorical_cols.append(col)
        elif is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def make_tree_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("to_string", FunctionTransformer(lambda x: x.astype(str), validate=False)),
        ("onehot", make_onehot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def make_catboost_frame(X: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    Xc = X.copy()
    for col in numeric_cols:
        Xc[col] = pd.to_numeric(Xc[col], errors="coerce")
        Xc[col] = Xc[col].fillna(Xc[col].median())
    for col in categorical_cols:
        Xc[col] = Xc[col].astype("object").where(Xc[col].notna(), "Missing").astype(str)
    return Xc


def check_required_libraries() -> None:
    missing = []
    if XGBClassifier is None:
        missing.append(f"xgboost ({XGB_IMPORT_ERROR})")
    if LGBMClassifier is None:
        missing.append(f"lightgbm ({LGBM_IMPORT_ERROR})")
    if CatBoostClassifier is None:
        missing.append(f"catboost ({CATBOOST_IMPORT_ERROR})")
    if missing:
        print("Missing required packages for full reproduction:")
        for m in missing:
            print(" -", m)
        print("\nInstall them with:")
        print("pip install xgboost lightgbm catboost")
        sys.exit(1)


def get_base_models() -> Dict[str, Any]:
    """Base model parameters used to generate the optimized ensemble predictions."""
    return {
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.02,
            reg_lambda=2.5,
            reg_alpha=0.08,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=650,
            learning_rate=0.022,
            num_leaves=24,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.08,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=650,
            depth=5,
            learning_rate=0.025,
            l2_leaf_reg=5.0,
            random_strength=0.8,
            bagging_temperature=0.6,
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).values


def blend_base_predictions(base_probs: pd.DataFrame) -> np.ndarray:
    """Apply the fixed best static ensemble formula."""
    required = ["xgboost", "lightgbm", "catboost"]
    missing = [c for c in required if c not in base_probs.columns]
    if missing:
        raise ValueError(f"Base prediction file is missing columns: {missing}")

    x = base_probs["xgboost"].values
    l = base_probs["lightgbm"].values
    c = base_probs["catboost"].values

    prob_blend = W_XGBOOST * x + W_LIGHTGBM * l + W_CATBOOST * c

    rank_blend = (
        W_XGBOOST * percentile_rank(x)
        + W_LIGHTGBM * percentile_rank(l)
        + W_CATBOOST * percentile_rank(c)
    )

    final_score = ALPHA_PROBABILITY * prob_blend + (1.0 - ALPHA_PROBABILITY) * rank_blend
    return final_score


def train_base_predictions(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> pd.DataFrame:
    """Train 5-fold base models and average their test probabilities."""
    check_required_libraries()
    numeric_cols, categorical_cols = get_feature_columns(X)
    models = get_base_models()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    test_probs: Dict[str, np.ndarray] = {name: np.zeros(len(X_test)) for name in models}

    for model_name, base_model in models.items():
        log(f"Training base model: {model_name}")
        for fold, (tr_idx, _val_idx) in enumerate(skf.split(X, y), start=1):
            log(f"  {model_name} | fold {fold}/{N_SPLITS}")
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]

            if model_name == "catboost":
                model = clone(base_model)
                X_tr_cb = make_catboost_frame(X_tr, numeric_cols, categorical_cols)
                X_test_cb = make_catboost_frame(X_test, numeric_cols, categorical_cols)
                cat_feature_indices = [X_tr_cb.columns.get_loc(c) for c in categorical_cols]
                model.fit(X_tr_cb, y_tr, cat_features=cat_feature_indices)
                test_prob = model.predict_proba(X_test_cb)[:, 1]
            else:
                model = clone(base_model)
                preprocessor = make_tree_preprocessor(numeric_cols, categorical_cols)
                X_tr_enc = preprocessor.fit_transform(X_tr)
                X_test_enc = preprocessor.transform(X_test)
                model.fit(X_tr_enc, y_tr)
                test_prob = model.predict_proba(X_test_enc)[:, 1]

            test_probs[model_name] += test_prob / N_SPLITS

    return pd.DataFrame(test_probs)



def save_submission(test_ids: pd.Series, final_score: np.ndarray, output_dir: Path, sample_path: str | None) -> str:
    pred_bool = final_score >= DECISION_THRESHOLD

    if sample_path and Path(sample_path).exists():
        submission = pd.read_csv(sample_path)
        if "PassengerId" in submission.columns and len(submission) == len(test_ids):
            submission["PassengerId"] = test_ids.values
        else:
            submission = pd.DataFrame({"PassengerId": test_ids.values})

        # Spaceship Titanic expects Transported as True/False.
        if "Transported" in submission.columns:
            submission["Transported"] = pred_bool.astype(bool)
        elif "Survived" in submission.columns:
            submission["Survived"] = pred_bool.astype(int)
        else:
            submission["Transported"] = pred_bool.astype(bool)
    else:
        submission = pd.DataFrame({"PassengerId": test_ids.values, "Transported": pred_bool.astype(bool)})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_final_submission_retrain_only.csv"
    submission.to_csv(out_path, index=False)
    return str(out_path)


def save_parameters(output_dir: Path) -> str:
    params = pd.DataFrame([
        {"parameter": "w_xgboost", "value": W_XGBOOST},
        {"parameter": "w_lightgbm", "value": W_LIGHTGBM},
        {"parameter": "w_catboost", "value": W_CATBOOST},
        {"parameter": "alpha_probability", "value": ALPHA_PROBABILITY},
        {"parameter": "decision_threshold", "value": DECISION_THRESHOLD},
        {"parameter": "n_splits", "value": N_SPLITS},
        {"parameter": "best_kaggle_score_observed", "value": 0.81201},
    ])
    out_path = output_dir / "demo_final_parameters.csv"
    params.to_csv(out_path, index=False)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None, help="Path to train.csv")
    parser.add_argument("--test", default=None, help="Path to test.csv")
    parser.add_argument("--submission", default=None, help="Path to sample_submission.csv")
    parser.add_argument("--output-dir", default="demo_final_outputs", help="Output folder")
    args = parser.parse_args()

    train_path = resolve_path(args.train, ["train.csv", "train(4).csv"], required=True)
    test_path = resolve_path(args.test, ["test.csv", "test(4).csv"], required=True)
    sample_path = resolve_path(
        args.submission,
        ["sample_submission.csv", "sample_submission(3).csv", "gender_submission.csv"],
        required=False,
    )

    output_dir = Path(args.output_dir).resolve()
    log(f"Train path: {train_path}")
    log(f"Test path:  {test_path}")
    log(f"Output dir: {output_dir}")
    log("Using fixed final ensemble parameters:")
    log(f"  XGB={W_XGBOOST}, LGBM={W_LIGHTGBM}, CAT={W_CATBOOST}, alpha={ALPHA_PROBABILITY}, threshold={DECISION_THRESHOLD}")

    X, y, X_test, test_ids = prepare_data(train_path, test_path)
    log(f"Train features shape: {X.shape}")
    log(f"Test features shape:  {X_test.shape}")

    # This retrain-only version never reads previous prediction files.
    # It always trains base models from train.csv and predicts test.csv.
    log("Retrain-only mode: training base models now. No previous prediction file will be read.")
    base_probs = train_base_predictions(X, y, X_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_probs.to_csv(output_dir / "demo_test_base_predictions_retrained.csv", index=False)
    log(f"Saved newly trained base predictions: {output_dir / 'demo_test_base_predictions_retrained.csv'}")

    if len(base_probs) != len(test_ids):
        raise ValueError(
            f"Base predictions length ({len(base_probs)}) does not match test length ({len(test_ids)}). "
            "Please check that train.csv and test.csv are the correct Spaceship Titanic files."
        )

    final_score = blend_base_predictions(base_probs)
    submission_path = save_submission(test_ids, final_score, output_dir, sample_path)
    params_path = save_parameters(output_dir)

    log(f"Saved final submission: {submission_path}")
    log(f"Saved fixed parameters: {params_path}")
    log("Done.")


if __name__ == "__main__":
    main()
