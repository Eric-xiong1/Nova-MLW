"""
Spaceship Titanic final demo pipeline - simplified same-folder version
====================================================================

Put this .py file in the same folder as:
- train.csv
- test.csv
- sample_submission.csv  (optional, not required by this script)

Then run:
python spaceship_demo_code_simplified_same_folder.py

The script trains XGBoost, LightGBM, and CatBoost from the original train.csv/test.csv,
uses 5-fold OOF validation, blends model predictions, and saves a Kaggle submission file.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


RANDOM_STATE = 42
N_SPLITS = 5

# Final fixed ensemble settings
## Integrated model weights (CatBoost has the highest weight and serves as the core model)
W_XGBOOST = 0.0075
W_LIGHTGBM = 0.2325
W_CATBOOST = 0.7600
ALPHA_PROBABILITY = 0.8125 # Probability Fusion Weight (Probability + Ranking)
DECISION_THRESHOLD = 0.5050 ## Classification Threshold (Samples with value >0.505 are judged as Transported)


# Fixed same-folder paths
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR / "train.csv"
TEST_PATH = SCRIPT_DIR / "test.csv"
OUTPUT_DIR = SCRIPT_DIR / "demo_final_outputs"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful Spaceship Titanic features from the original columns."""
    df = df.copy()
    # 1. Split PassengerId into GroupId + PassengerNo
    split_id = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["GroupId"] = split_id[0]
    df["PassengerNo"] = pd.to_numeric(split_id[1], errors="coerce")
    # 2. Cabin Split: Deck + Num + Side
    cabin_split = df["Cabin"].astype(str).replace("nan", np.nan).str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]
    # 3. Name Features: Surname + Name Length
    df["Surname"] = df["Name"].astype(str).str.split().str[-1]
    df.loc[df["Name"].isna(), "Surname"] = np.nan
    df["NameLength"] = df["Name"].astype(str).str.len()
    # 4. Consumption Characteristics
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1, skipna=True)
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)
    df["LuxurySpend"] = df[["FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1, skipna=True)
    df["BasicSpend"] = df["RoomService"].fillna(0)
    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1)

    for col in spend_cols + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
        df[f"Log_{col}"] = np.log1p(df[col].fillna(0))
    # 5. Age Characteristics: Binning
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 30, 45, 60, 120],
        labels=["child", "teen", "young", "adult", "middle", "senior"],
    ).astype("object")
    df["IsChild"] = ((df["Age"].notna()) & (df["Age"] < 13)).astype(int)
    df["IsSenior"] = ((df["Age"].notna()) & (df["Age"] >= 60)).astype(int)
    ## 6. Interactive Features: CryoSleep + Spending (People in cryosleep generally make no purchases)
    cryo_true = df["CryoSleep"].astype(str).str.lower().eq("true")
    df["CryoAndSpend"] = (cryo_true & (df["TotalSpend"].fillna(0) > 0)).astype(int)
    df["CryoAndNoSpend"] = (cryo_true & (df["TotalSpend"].fillna(0) == 0)).astype(int)

    return df


def add_count_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add group and surname count features using non-label information only."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["__is_train__"] = 1
    test_df["__is_train__"] = 0

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined["GroupSize"] = combined.groupby("GroupId")["PassengerId"].transform("count")
    combined["IsSolo"] = (combined["GroupSize"] == 1).astype(int)
    combined["SurnameSize"] = combined.groupby("Surname")["PassengerId"].transform("count")

    train_new = combined[combined["__is_train__"] == 1].drop(columns=["__is_train__"])
    test_new = combined[combined["__is_train__"] == 0].drop(columns=["__is_train__"])
    return train_new.reset_index(drop=True), test_new.reset_index(drop=True)


def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Read train.csv/test.csv from the same folder and prepare features."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y = train_df["Transported"].astype(bool).astype(int)
    test_ids = test_df["PassengerId"].copy()

    train_features = add_features(train_df.drop(columns=["Transported"]))
    test_features = add_features(test_df)
    train_features, test_features = add_count_features(train_features, test_features)
    ## Delete the original ID, name and cabin columns (replaced by feature engineering)
    drop_cols = ["PassengerId", "Name", "Cabin"]
    train_features = train_features.drop(columns=drop_cols)
    test_features = test_features.drop(columns=drop_cols)

    return train_features, y, test_features, test_ids


def get_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separate numerical and categorical columns."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols

#data preparation for the model
def make_tree_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Preprocess data for XGBoost and LightGBM."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("to_string", FunctionTransformer(lambda x: x.astype(str), validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def make_catboost_frame(X: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """Prepare data for CatBoost with native categorical handling."""
    Xc = X.copy()
    for col in numeric_cols:
        Xc[col] = pd.to_numeric(Xc[col], errors="coerce")
        Xc[col] = Xc[col].fillna(Xc[col].median())
    for col in categorical_cols:
        Xc[col] = Xc[col].astype("object").where(Xc[col].notna(), "Missing").astype(str)
    return Xc


def get_base_models() -> Dict[str, Any]:
    """Define the final selected model configurations."""
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
    """Blend model probabilities and percentile ranks."""
    x = base_probs["xgboost"].values
    l = base_probs["lightgbm"].values
    c = base_probs["catboost"].values
    # 1. Probability Fusion: Weighted Summation
    prob_blend = W_XGBOOST * x + W_LIGHTGBM * l + W_CATBOOST * c
    # 2. Rank Fusion: Convert probabilities to percentile ranks and then apply weighted summation
    rank_blend = (
        W_XGBOOST * percentile_rank(x)
        + W_LIGHTGBM * percentile_rank(l)
        + W_CATBOOST * percentile_rank(c)
    )
    # 3. Final Fusion
    return ALPHA_PROBABILITY * prob_blend + (1.0 - ALPHA_PROBABILITY) * rank_blend


def train_base_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train base models with 5-fold StratifiedKFold."""
    numeric_cols, categorical_cols = get_feature_columns(X)
    models = get_base_models()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # Initialize containers for test set probabilities and OOF probabilities
    test_probs: Dict[str, np.ndarray] = {name: np.zeros(len(X_test)) for name in models}
    oof_probs: Dict[str, np.ndarray] = {name: np.zeros(len(X)) for name in models}
    fold_records: List[Dict[str, Any]] = []
    runtime_records: List[Dict[str, Any]] = []

    for model_name, base_model in models.items():
        log(f"Training base model: {model_name}")
        model_start = time.time()

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            fold_start = time.time()
            log(f"  {model_name} | fold {fold}/{N_SPLITS}")
            # Split Training/Validation Datasets
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = clone(base_model)
            # Preprocessing Logic of Different Models
            if model_name == "catboost":
                X_tr_cb = make_catboost_frame(X_tr, numeric_cols, categorical_cols)
                X_val_cb = make_catboost_frame(X_val, numeric_cols, categorical_cols)
                X_test_cb = make_catboost_frame(X_test, numeric_cols, categorical_cols)
                cat_feature_indices = [X_tr_cb.columns.get_loc(c) for c in categorical_cols]

                model.fit(X_tr_cb, y_tr, cat_features=cat_feature_indices)
                val_prob = model.predict_proba(X_val_cb)[:, 1]
                test_prob = model.predict_proba(X_test_cb)[:, 1]
            else:
                preprocessor = make_tree_preprocessor(numeric_cols, categorical_cols)
                X_tr_enc = preprocessor.fit_transform(X_tr)
                X_val_enc = preprocessor.transform(X_val)
                X_test_enc = preprocessor.transform(X_test)

                model.fit(X_tr_enc, y_tr)
                val_prob = model.predict_proba(X_val_enc)[:, 1]
                test_prob = model.predict_proba(X_test_enc)[:, 1]

            oof_probs[model_name][val_idx] = val_prob
            test_probs[model_name] += test_prob / N_SPLITS

            fold_time = time.time() - fold_start
            fold_acc = accuracy_score(y_val, val_prob >= 0.5)
            fold_records.append({
                "model": model_name,
                "fold": fold,
                "fold_accuracy_at_0_5": fold_acc,
                "fold_time_seconds": fold_time,
            })
            log(f"    fold accuracy: {fold_acc:.5f} | time: {fold_time:.2f}s")

        model_time = time.time() - model_start
        model_oof_acc = accuracy_score(y, oof_probs[model_name] >= 0.5)
        runtime_records.append({
            "model": model_name,
            "oof_accuracy_at_0_5": model_oof_acc,
            "training_time_seconds": model_time,
            "n_splits": N_SPLITS,
        })
        log(f"  {model_name} OOF accuracy: {model_oof_acc:.5f} | total time: {model_time:.2f}s")

    return (
        pd.DataFrame(test_probs),
        pd.DataFrame(oof_probs),
        pd.DataFrame(fold_records),
        pd.DataFrame(runtime_records),
    )


def save_outputs(
    test_ids: pd.Series,
    base_probs: pd.DataFrame,
    oof_probs: pd.DataFrame,
    fold_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Save submission and validation files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_oof_score = blend_base_predictions(oof_probs)
    final_oof_pred = final_oof_score >= DECISION_THRESHOLD
    final_oof_accuracy = accuracy_score(y, final_oof_pred)

    final_test_score = blend_base_predictions(base_probs)
    final_test_pred = final_test_score >= DECISION_THRESHOLD

    submission = pd.DataFrame({
        "PassengerId": test_ids.values,
        "Transported": final_test_pred.astype(bool),
    })

    validation_records = []
    for model_name in oof_probs.columns:
        validation_records.append({
            "model": model_name,
            "validation_type": f"{N_SPLITS}-fold OOF",
            "threshold": 0.5,
            "accuracy": accuracy_score(y, oof_probs[model_name] >= 0.5),
        })
    validation_records.append({
        "model": "final_ensemble",
        "validation_type": f"{N_SPLITS}-fold OOF",
        "threshold": DECISION_THRESHOLD,
        "accuracy": final_oof_accuracy,
    })

    oof_output = oof_probs.copy()
    oof_output["final_ensemble_score"] = final_oof_score
    oof_output["final_ensemble_prediction"] = final_oof_pred.astype(bool)
    oof_output["true_Transported"] = y.astype(bool).values

    base_probs.to_csv(OUTPUT_DIR / "demo_test_base_predictions.csv", index=False)
    submission.to_csv(OUTPUT_DIR / "demo_final_submission.csv", index=False)
    pd.DataFrame(validation_records).to_csv(OUTPUT_DIR / "demo_validation_summary.csv", index=False)
    fold_summary.to_csv(OUTPUT_DIR / "demo_fold_validation_summary.csv", index=False)
    runtime_summary.to_csv(OUTPUT_DIR / "demo_model_runtime_summary.csv", index=False)
    oof_output.to_csv(OUTPUT_DIR / "demo_oof_predictions.csv", index=False)

    log(f"Final ensemble OOF accuracy {DECISION_THRESHOLD}: {final_oof_accuracy:.5f}")
    log(f"Saved final submission: {OUTPUT_DIR / 'demo_final_submission.csv'}")
    log(f"Saved validation summary: {OUTPUT_DIR / 'demo_validation_summary.csv'}")
    log(f"Saved runtime summary: {OUTPUT_DIR / 'demo_model_runtime_summary.csv'}")


def main() -> None:
    log(f"Train path: {TRAIN_PATH}")
    log(f"Test path:  {TEST_PATH}")
    log(f"Output dir: {OUTPUT_DIR}")

    X, y, X_test, test_ids = prepare_data()
    log(f"Train features shape: {X.shape}")
    log(f"Test features shape:  {X_test.shape}")

    base_probs, oof_probs, fold_summary, runtime_summary = train_base_predictions(X, y, X_test)
    save_outputs(test_ids, base_probs, oof_probs, fold_summary, runtime_summary, y)
    log("Done.")


if __name__ == "__main__":
    main()
