# ============================================================
# 2_train_models.py
# Compare 5 ML models with Optuna (5-fold CV), evaluate on a
# fixed 20% holdout set, then save the final best predictive
# model for future CE prediction.
# ============================================================

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt

from optuna.visualization.matplotlib import plot_optimization_history

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = Path("cleaned_final_features.csv")
TARGET_COLUMN = "CE_aver. (%)"
SMILES_COLUMN = "Additive_SMILES"

# Final manuscript run: 40 Optuna trials/model + 5-fold inner CV.
QUICK_RUN = False
TEST_SIZE = 0.20
RANDOM_STATE = 42

MODEL_NAMES = ["XGBoost", "LightGBM", "HistGradientBoosting", "ExtraTrees", "RandomForest"]

if QUICK_RUN:
    N_OPTUNA_TRIALS = 5
    INNER_CV_FOLDS = 3
else:
    N_OPTUNA_TRIALS = 40
    INNER_CV_FOLDS = 5

SAVED_MODEL_DIR = Path("saved_models")
VIS_DIR = Path("visualizations")
RESULTS_DIR = Path("predictive ML performance metrics")
for folder in (SAVED_MODEL_DIR, VIS_DIR, RESULTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})

# ============================================================
# 2. HELPERS
# ============================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.20, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 15.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 10, 15]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        }
    if model_name == "HistGradientBoosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.30, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 20.0, log=True),
        }
    if model_name == "ExtraTrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 20, 30, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.30, 0.50, 0.80, 1.0]),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 20, 30, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.30, 0.50, 0.80, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name: str, params: dict[str, Any]) -> Pipeline:
    # Save preprocessing + estimator together in one pipeline so the exact
    # preprocessing is reused automatically for future CE predictions.
    preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("remove_constant_features", VarianceThreshold()),
    ]

    if model_name == "XGBoost":
        estimator = XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=1, **params
        )
    elif model_name == "LightGBM":
        estimator = LGBMRegressor(
            objective="regression", random_state=RANDOM_STATE, n_jobs=1,
            verbosity=-1, **params
        )
    elif model_name == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    elif model_name == "ExtraTrees":
        estimator = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(preprocessing + [("model", estimator)])


# ============================================================
# 3. MAIN
# ============================================================

def main():
    print("=" * 76)
    print("5-MODEL CE TRAINING + FINAL PREDICTIVE MODEL")
    print("=" * 76)
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)
    required = {TARGET_COLUMN, SMILES_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[TARGET_COLUMN, SMILES_COLUMN]).copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    LEAKAGE_COLUMNS = [
        TARGET_COLUMN, "CE_1 (%)", "CE_2 (%)", "CE_3 (%)",
        "CE_average (%)", "CE_aver. (%)", "LCE", "LogCE",
    ]
    IDENTIFIER_COLUMNS = [
        SMILES_COLUMN, "#", "IUPAC_NAME", "id", "ID", "Name",
        "Additive_Name", "Compound_Name",
    ]

    exclude_columns = [c for c in LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS if c in df.columns]
    X = df.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    y = df[TARGET_COLUMN].astype(float).copy()

    if X.empty:
        raise ValueError("No numeric predictive features remain.")

    print(f"Rows/formulations: {len(X)}")
    print(f"Unique additives: {df[SMILES_COLUMN].nunique()}")
    print(f"Numeric predictors: {X.shape[1]}")
    print(f"Optuna folds: {INNER_CV_FOLDS}")
    print(f"Optuna trials/model: {N_OPTUNA_TRIALS}")
    print(f"Holdout split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")

    # Save the exact columns required later for prediction.
    joblib.dump(list(X.columns), SAVED_MODEL_DIR / "model_feature_columns.joblib")

    # ------------------------------------------------------------
    # Fixed 80/20 split for fair comparison of all five models.
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )

    inner_cv = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    comparison_rows = []
    best_params_by_model = {}
    best_models = {}
    studies = {}

    # ============================================================
    # 4. OPTUNA TUNING: 5 MODELS, 5-FOLD CV
    # ============================================================
    for model_index, model_name in enumerate(MODEL_NAMES):
        print("\n" + "=" * 76)
        print(f"TUNING {model_name}")
        print("=" * 76)

        def objective(trial):
            params = suggest_parameters(trial, model_name)
            model = build_model(model_name, params)
            scores = cross_val_score(
                model, X_train, y_train, cv=inner_cv,
                scoring="r2", n_jobs=-1, error_score="raise"
            )
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + model_index),
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        studies[model_name] = study
        best_params = study.best_params
        best_params_by_model[model_name] = best_params

        # Final evaluation model for this algorithm: fit on all 80% train data.
        model = build_model(model_name, best_params)
        model.fit(X_train, y_train)
        best_models[model_name] = model

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        comparison_rows.append({
            "model": model_name,
            "best_inner_cv_r2": study.best_value,
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_rmse": rmse(y_train, train_pred),
            "test_rmse": rmse(y_test, test_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
        })

        study.trials_dataframe().to_csv(RESULTS_DIR / f"{model_name}_optuna_trials.csv", index=False)

        fig = plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{model_name}_optuna_history.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Best inner-CV R²: {study.best_value:.4f}")
        print(f"Train R²: {comparison_rows[-1]['train_r2']:.4f}")
        print(f"Test R²:  {comparison_rows[-1]['test_r2']:.4f}")

    # ============================================================
    # 5. SAVE COMPARISON + BEST HYPERPARAMETERS
    # ============================================================
    comparison = pd.DataFrame(comparison_rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    comparison.to_csv(RESULTS_DIR / "train_test_model_comparison.csv", index=False)

    with open(RESULTS_DIR / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params_by_model, f, indent=2)

    print("\nFINAL HOLDOUT COMPARISON")
    print("=" * 100)
    print(comparison.round(4).to_string(index=False))

    # Preserve previous selection logic: highest fixed-holdout test R² wins.
    winner_name = str(comparison.iloc[0]["model"])
    winner_test_r2 = float(comparison.iloc[0]["test_r2"])
    winner_params = best_params_by_model[winner_name]
    holdout_winner = best_models[winner_name]

    print("\n" + "=" * 76)
    print("SELECTED BEST MODEL")
    print("=" * 76)
    print(f"Best model: {winner_name}")
    print(f"Holdout test R²: {winner_test_r2:.4f}")

    # Save the evaluated winner (fit only on the 80% training partition).
    joblib.dump(holdout_winner, SAVED_MODEL_DIR / "best_model_80pct_holdout_evaluated.joblib")

    # ============================================================
    # 6. VISUALIZATIONS (same core outputs as old XGB script)
    # ============================================================
    y_pred = holdout_winner.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.65, edgecolors="black", linewidths=0.5)
    finite = np.concatenate([np.asarray(y_test, dtype=float), np.asarray(y_pred, dtype=float)])
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = max((high - low) * 0.05, 0.5)
    ax.plot([low-pad, high+pad], [low-pad, high+pad], "r--", alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(low-pad, high+pad)
    ax.set_ylim(low-pad, high+pad)
    ax.set_title(f"{winner_name}: Predicted vs Actual Coulombic Efficiency", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual CE (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted CE (%)", fontsize=12, fontweight="bold")
    ax.text(
        0.05, 0.95, f"$R^2 = {winner_test_r2:.4f}$", transform=ax.transAxes,
        fontsize=14, va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray")
    )
    plt.tight_layout()
    plt.savefig(VIS_DIR / "predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Keep the old generic filename for the selected winner's Optuna history.
    fig = plot_optimization_history(studies[winner_name])
    plt.tight_layout()
    plt.savefig(VIS_DIR / "optuna_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    x = np.arange(len(comparison))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, comparison["train_r2"], width, label="Train R²")
    ax.bar(x + width/2, comparison["test_r2"], width, label="Test R²")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["model"], rotation=25, ha="right")
    ax.set_ylabel("R²")
    ax.set_xlabel("Model")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "train_vs_test_R2.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 7. FINAL DEPLOYMENT MODEL: SAME WINNER + SAME PARAMS, 100% DATA
    # ============================================================
    # Optuna is NOT rerun here. The already-selected best algorithm and
    # hyperparameters are reused, then fitted once on all available data.
    final_model = build_model(winner_name, winner_params)
    final_model.fit(X, y)

    joblib.dump(final_model, SAVED_MODEL_DIR / "final_best_model.joblib")
    safe_name = winner_name.lower().replace(" ", "_")
    joblib.dump(final_model, SAVED_MODEL_DIR / f"final_{safe_name}_model.joblib")

    # Save preprocessing separately as well, to preserve the old workflow output.
    fitted_preprocessor = Pipeline(final_model.steps[:-1])
    joblib.dump(fitted_preprocessor, SAVED_MODEL_DIR / "preprocessor.joblib")

    # Also save the optimized XGBoost pipeline under the old familiar filename.
    # If XGBoost is not the overall winner, this is still the optimized XGBoost
    # from the same 5-model comparison, fitted on 100% of the data.
    final_xgb_pipeline = build_model("XGBoost", best_params_by_model["XGBoost"])
    final_xgb_pipeline.fit(X, y)
    joblib.dump(final_xgb_pipeline, SAVED_MODEL_DIR / "xgb_best_model.joblib")

    deployment_info = {
        "selected_best_model": winner_name,
        "selection_metric": "highest fixed-holdout test R2",
        "holdout_test_r2": winner_test_r2,
        "holdout_test_size": TEST_SIZE,
        "optuna_trials_per_model": N_OPTUNA_TRIALS,
        "inner_cv_folds": INNER_CV_FOLDS,
        "random_state": RANDOM_STATE,
        "target_column": TARGET_COLUMN,
        "number_of_training_rows_full": int(len(X)),
        "number_of_input_features_before_pipeline_filtering": int(X.shape[1]),
        "best_hyperparameters": winner_params,
        "final_model_file": "final_best_model.joblib",
        "feature_columns_file": "model_feature_columns.joblib",
    }
    with open(SAVED_MODEL_DIR / "final_model_info.json", "w", encoding="utf-8") as f:
        json.dump(deployment_info, f, indent=2)

    print("\n" + "=" * 76)
    print("TRAINING COMPLETE")
    print("=" * 76)
    print(f"Selected best model: {winner_name}")
    print(f"Reported holdout test R²: {winner_test_r2:.4f}")
    print(f"Final deployment model: {SAVED_MODEL_DIR / 'final_best_predictive_model.joblib'}")
    print(f"Feature schema: {SAVED_MODEL_DIR / 'model_feature_columns.joblib'}")
    print(f"Compatibility preprocessor: {SAVED_MODEL_DIR / 'preprocessor.joblib'}")
    print(f"Optimized XGBoost pipeline: {SAVED_MODEL_DIR / 'xgb_best_model.joblib'}")
    print(f"Predicted-vs-actual plot: {VIS_DIR / 'predicted_vs_actual.png'}")
    print(f"Winner Optuna history: {VIS_DIR / 'optuna_history.png'}")
    print("\nIMPORTANT:")
    print("The reported test R² is from the 80/20 evaluation model.")
    print("The final deployment model is then refit on 100% of the data using")
    print("the already-selected winning algorithm and hyperparameters.")


if __name__ == "__main__":
    main()
