# ============================================================
# 2_train_models.py
# Compare 5 ML models with Optuna (5-fold CV), evaluate on a
# fixed 20% holdout set, generate manuscript-ready plots for all
# models, then save the final best predictive model for future
# CE prediction and recursive discovery.
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
MODEL_DISPLAY_NAMES = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "HistGradientBoosting": "HistoBoost",
    "ExtraTrees": "ExtraTrees",
    "RandomForest": "Random Forest",
}

if QUICK_RUN:
    N_OPTUNA_TRIALS = 5
    INNER_CV_FOLDS = 3
else:
    N_OPTUNA_TRIALS = 40
    INNER_CV_FOLDS = 5

SAVED_MODEL_DIR = Path("saved_models")
VIS_DIR = Path("visualizations")
RESULTS_DIR = Path("predictive ML performance metrics")
PREDICTIONS_DIR = RESULTS_DIR / "per_model_predictions"
for folder in (SAVED_MODEL_DIR, VIS_DIR, RESULTS_DIR, PREDICTIONS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Canonical outputs consumed by the recursive discovery workflow.
FINAL_MODEL_PATH = SAVED_MODEL_DIR / "final_best_model.joblib"
FEATURE_COLUMNS_PATH = SAVED_MODEL_DIR / "model_feature_columns.joblib"

# Plot colors sampled from the user-provided benchmark figure.
AMPERE_BLUE = "#2051CE"
BENCHMARK_GREEN = "#77DECB"
OTHER_GREY = "#DEDEDE"
REFERENCE_LINE_COLOR = "#222222"
GRID_COLOR = "#BDBDBD"

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

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


def save_prediction_table(model_name: str, y_train, train_pred, y_test, test_pred) -> None:
    output = pd.DataFrame({
        "split": ["train"] * len(y_train) + ["test"] * len(y_test),
        "experimental_ce": np.concatenate([np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)]),
        "predicted_ce": np.concatenate([np.asarray(train_pred, dtype=float), np.asarray(test_pred, dtype=float)]),
    })
    output.to_csv(PREDICTIONS_DIR / f"{model_name}_predictions.csv", index=False)


def _style_axis(ax):
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45, color=GRID_COLOR)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_parity_plot(ax, model_name: str, y_train, train_pred, y_test, test_pred, metrics: dict[str, float], add_panel_label: str | None = None):
    y_train = np.asarray(y_train, dtype=float)
    train_pred = np.asarray(train_pred, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)

    combined = np.concatenate([y_train, train_pred, y_test, test_pred])
    finite = combined[np.isfinite(combined)]
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = max((high - low) * 0.05, 0.5)

    ax.scatter(y_train, train_pred, s=28, alpha=0.65, color=AMPERE_BLUE, label="Train", edgecolors="none")
    ax.scatter(y_test, test_pred, s=28, alpha=0.80, color=BENCHMARK_GREEN, label="Test", edgecolors="none")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", linewidth=1.4, color=REFERENCE_LINE_COLOR)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental CE (%)", fontweight="bold")
    ax.set_ylabel("Predicted CE (%)", fontweight="bold")
    ax.set_title(MODEL_DISPLAY_NAMES.get(model_name, model_name), fontsize=15, fontweight="bold")
    _style_axis(ax)
    ax.legend(frameon=True, fontsize=9)

    metrics_text = (
        f"Train R² = {metrics['train_r2']:.2f}\n"
        f"Test R² = {metrics['test_r2']:.2f}\n"
        f"Test RMSE = {metrics['test_rmse']:.2f}\n"
        f"Test MAE = {metrics['test_mae']:.2f}"
    )
    ax.text(
        0.98,
        0.04,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#666666", alpha=0.95),
    )
    if add_panel_label:
        ax.text(
            -0.12,
            1.05,
            add_panel_label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
        )


def make_individual_parity_plots(predictions_by_model: dict[str, dict[str, np.ndarray]], comparison_df: pd.DataFrame) -> None:
    for _, row in comparison_df.iterrows():
        model_name = str(row["model"])
        metrics = row.to_dict()
        pred = predictions_by_model[model_name]
        fig, ax = plt.subplots(figsize=(8, 8))
        create_parity_plot(
            ax,
            model_name,
            pred["y_train"],
            pred["train_pred"],
            pred["y_test"],
            pred["test_pred"],
            metrics,
        )
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{model_name}_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_r2_bar_plot(comparison_df: pd.DataFrame) -> None:
    ordered = comparison_df.copy()
    display_names = [MODEL_DISPLAY_NAMES.get(name, name) for name in ordered["model"]]
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    train_bars = ax.bar(x - width / 2, ordered["train_r2"], width, label="Train R²", color=AMPERE_BLUE)
    test_bars = ax.bar(x + width / 2, ordered["test_r2"], width, label="Test R²", color=BENCHMARK_GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=28, ha="right", fontweight="bold")
    ax.set_ylabel("R² Score", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylim(0, max(1.15, float(np.nanmax(ordered[["train_r2", "test_r2"]].to_numpy())) + 0.08))
    _style_axis(ax)
    ax.legend(frameon=True)

    for bar in list(train_bars) + list(test_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(VIS_DIR / "train_vs_test_R2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_metric_bar_plots(comparison_df: pd.DataFrame) -> None:
    ordered = comparison_df.copy()
    display_names = [MODEL_DISPLAY_NAMES.get(name, name) for name in ordered["model"]]
    x = np.arange(len(ordered))
    width = 0.36

    metric_specs = [
        ("test_r2", "Test R²", AMPERE_BLUE, "higher"),
        ("test_rmse", "Test RMSE", BENCHMARK_GREEN, "lower"),
        ("test_mae", "Test MAE", OTHER_GREY, "lower"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, (column, title, color, _) in zip(axes, metric_specs):
        bars = ax.bar(x, ordered[column], color=color, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=28, ha="right", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Model", fontweight="bold")
        _style_axis(ax)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.01 if column == "test_r2" else 0.05),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        if column == "test_r2":
            ax.set_ylabel("R² Score", fontweight="bold")
        else:
            ax.set_ylabel("Error (% CE)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_metric_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_summary_figure(predictions_by_model: dict[str, dict[str, np.ndarray]], comparison_df: pd.DataFrame) -> None:
    panel_order = ["XGBoost", "LightGBM", "HistGradientBoosting", "ExtraTrees", "RandomForest"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    comparison_lookup = {str(row["model"]): row.to_dict() for _, row in comparison_df.iterrows()}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2))
    axes = axes.ravel()

    for ax, model_name, panel_label in zip(axes[:5], panel_order, panel_labels):
        pred = predictions_by_model[model_name]
        create_parity_plot(
            ax,
            model_name,
            pred["y_train"],
            pred["train_pred"],
            pred["y_test"],
            pred["test_pred"],
            comparison_lookup[model_name],
            add_panel_label=panel_label,
        )

    # Sixth panel: train vs test R² bar plot.
    ax = axes[5]
    ordered = comparison_df.set_index("model").loc[panel_order].reset_index()
    display_names = [MODEL_DISPLAY_NAMES.get(name, name) for name in ordered["model"]]
    x = np.arange(len(ordered))
    width = 0.38
    train_bars = ax.bar(x - width / 2, ordered["train_r2"], width, label="Train R²", color=AMPERE_BLUE)
    test_bars = ax.bar(x + width / 2, ordered["test_r2"], width, label="Test R²", color=BENCHMARK_GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=28, ha="right", fontweight="bold")
    ax.set_ylabel("R² Score", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_title("Model comparison", fontweight="bold")
    ax.set_ylim(0, max(1.15, float(np.nanmax(ordered[["train_r2", "test_r2"]].to_numpy())) + 0.08))
    _style_axis(ax)
    ax.legend(frameon=True)
    for bar in list(train_bars) + list(test_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.text(-0.12, 1.05, "(f)", transform=ax.transAxes, fontsize=20, fontweight="bold", va="top", ha="left")

    plt.tight_layout()
    plt.savefig(VIS_DIR / "all_models_summary_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)

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
    predictions_by_model = {}

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

        row = {
            "model": model_name,
            "best_inner_cv_r2": study.best_value,
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_rmse": rmse(y_train, train_pred),
            "test_rmse": rmse(y_test, test_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
        }
        comparison_rows.append(row)

        predictions_by_model[model_name] = {
            "y_train": np.asarray(y_train, dtype=float),
            "train_pred": np.asarray(train_pred, dtype=float),
            "y_test": np.asarray(y_test, dtype=float),
            "test_pred": np.asarray(test_pred, dtype=float),
        }
        save_prediction_table(model_name, y_train, train_pred, y_test, test_pred)

        study.trials_dataframe().to_csv(RESULTS_DIR / f"{model_name}_optuna_trials.csv", index=False)

        fig = plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{model_name}_optuna_history.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Best inner-CV R²: {study.best_value:.4f}")
        print(f"Train R²: {row['train_r2']:.4f}")
        print(f"Test R²:  {row['test_r2']:.4f}")

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
    # 6. VISUALIZATIONS
    # ============================================================
    make_individual_parity_plots(predictions_by_model, comparison)
    make_r2_bar_plot(comparison)
    make_metric_bar_plots(comparison)
    make_combined_summary_figure(predictions_by_model, comparison)

    # Backward-compatible filenames for the selected winner.
    winner_fig_source = VIS_DIR / f"{winner_name}_predicted_vs_actual.png"
    if winner_fig_source.exists():
        import shutil
        shutil.copy2(winner_fig_source, VIS_DIR / "predicted_vs_actual.png")

    # Keep the old generic filename for the selected winner's Optuna history.
    fig = plot_optimization_history(studies[winner_name])
    plt.tight_layout()
    plt.savefig(VIS_DIR / "optuna_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 7. FINAL DEPLOYMENT MODEL: SAME WINNER + SAME PARAMS, 100% DATA
    # ============================================================
    # Optuna is NOT rerun here. The already-selected best algorithm and
    # hyperparameters are reused, then fitted once on all available data.
    final_model = build_model(winner_name, winner_params)
    final_model.fit(X, y)

    joblib.dump(final_model, FINAL_MODEL_PATH)
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
        "final_model_file": FINAL_MODEL_PATH.name,
        "feature_columns_file": FEATURE_COLUMNS_PATH.name,
        "comparison_metrics_file": "train_test_model_comparison.csv",
        "summary_figure_file": "all_models_summary_figure.png",
    }
    with open(SAVED_MODEL_DIR / "final_model_info.json", "w", encoding="utf-8") as f:
        json.dump(deployment_info, f, indent=2)

    print("\n" + "=" * 76)
    print("TRAINING COMPLETE")
    print("=" * 76)
    print(f"Selected best model: {winner_name}")
    print(f"Reported holdout test R²: {winner_test_r2:.4f}")
    print(f"Final deployment model: {FINAL_MODEL_PATH}")
    print(f"Feature schema: {FEATURE_COLUMNS_PATH}")
    print(f"Compatibility preprocessor: {SAVED_MODEL_DIR / 'preprocessor.joblib'}")
    print(f"Optimized XGBoost pipeline: {SAVED_MODEL_DIR / 'xgb_best_model.joblib'}")
    print(f"Winner parity plot: {VIS_DIR / 'predicted_vs_actual.png'}")
    print(f"All-model summary figure: {VIS_DIR / 'all_models_summary_figure.png'}")
    print(f"R² bar chart: {VIS_DIR / 'train_vs_test_R2.png'}")
    print(f"Test-metric summary chart: {VIS_DIR / 'test_metric_summary.png'}")
    print(f"Per-model prediction tables: {PREDICTIONS_DIR}")
    print(f"Winner Optuna history: {VIS_DIR / 'optuna_history.png'}")
    print("\nIMPORTANT:")
    print("The reported test R² is from the 80/20 evaluation model.")
    print("The final deployment model is then refit on 100% of the data using")
    print("the already-selected winning algorithm and hyperparameters.")


if __name__ == "__main__":
    main()
