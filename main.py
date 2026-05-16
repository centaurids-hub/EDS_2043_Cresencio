"""
Automated Water Quality Analysis and Anoxic Event Prediction
AQU-01: Anoxic Event Prediction | Pillar 3: Aquaculture & Hydrobiology
Author: Jesier S. Cresencio
Course: Computer Programming 1 | Final Project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy import stats
import warnings
import os
import time

warnings.filterwarnings("ignore")

# Change to the directory where main.py is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "data/filtered_dataset.csv"
OUTPUT_DIR  = "outputs"
ANOXIC_THRESH = 5.8        # mg/L  (near-anoxic threshold; yields ~50/50 class split)
FEATURES      = ["TEMP", "PH", "AMMONIA(mg/l)", "NITRATE(PPM)", "TURBIDITY"]
TARGET        = "anoxic"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def ingest_data(path: str) -> pd.DataFrame:
    """Load CSV dataset with error handling and initial inspection."""
    try:
        df = pd.read_csv(path)
        print(f"[INGESTION] Loaded {len(df)} records | Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Data ingestion failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, enforce correct data types."""
    try:
        initial_len = len(df)
        df = df.drop_duplicates()
        print(f"[CLEANING] Removed {initial_len - len(df)} duplicate records.")

        numeric_cols = ["NITRATE(PPM)", "PH", "AMMONIA(mg/l)", "TEMP", "DO",
                        "TURBIDITY", "MANGANESE(mg/l)"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        missing_before = df.isnull().sum().sum()
        df = df.dropna(subset=numeric_cols)
        print(f"[CLEANING] Dropped {missing_before} missing values. Records remaining: {len(df)}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — FEATURE ENGINEERING & TARGET CREATION
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, threshold: float = ANOXIC_THRESH) -> pd.DataFrame:
    """Create binary anoxic label and validate features."""
    try:
        df[TARGET] = (df["DO"] < threshold).astype(int)
        df["day_of_month"] = df["Date"].dt.day
        print(f"[FEATURES] Anoxic threshold: DO < {threshold} mg/L")
        print(f"[FEATURES] Anoxic=1: {df[TARGET].sum()} | Normal=0: {(df[TARGET]==0).sum()}")
        return df
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — STATISTICAL ANALYSIS (NumPy-powered)
# ─────────────────────────────────────────────────────────────────────────────
def statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics using NumPy.
    Returns a dictionary of metrics for each numeric feature + DO.
    """
    try:
        cols_of_interest = ["DO", "TEMP", "PH", "AMMONIA(mg/l)", "NITRATE(PPM)", "TURBIDITY"]
        stats_dict = {}
        print("\n" + "="*65)
        print(f"{'VARIABLE':<18} {'MEAN':>8} {'MEDIAN':>8} {'STD':>8} {'VAR':>10} {'SKEW':>7}")
        print("="*65)

        for col in cols_of_interest:
            arr = df[col].dropna().to_numpy()
            mean_val   = np.mean(arr)
            median_val = np.median(arr)
            std_val    = np.std(arr, ddof=1)
            var_val    = np.var(arr, ddof=1)
            skew_val   = stats.skew(arr)
            kurt_val   = stats.kurtosis(arr)

            stats_dict[col] = {
                "mean": mean_val, "median": median_val,
                "std": std_val,   "var": var_val,
                "skew": skew_val, "kurtosis": kurt_val,
                "min": np.min(arr), "max": np.max(arr),
                "q1": np.percentile(arr, 25), "q3": np.percentile(arr, 75)
            }
            print(f"{col:<18} {mean_val:>8.4f} {median_val:>8.4f} {std_val:>8.4f} {var_val:>10.4f} {skew_val:>7.4f}")

        print("="*65)

        # Correlation matrix using NumPy
        corr_cols = ["DO", "TEMP", "PH", "AMMONIA(mg/l)", "NITRATE(PPM)", "TURBIDITY"]
        corr_matrix = np.corrcoef(df[corr_cols].dropna().to_numpy().T)
        print("\n[STATS] Pearson Correlation (DO vs TEMP):",
              round(corr_matrix[0, 1], 4))

        return stats_dict

    except Exception as e:
        raise RuntimeError(f"Statistical analysis failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — STATIC VISUALIZATION  (≥3 chart types)
# ─────────────────────────────────────────────────────────────────────────────
def plot_static(df: pd.DataFrame):
    """Generate and save static visualizations."""
    try:
        palette = {"anoxic": "#e74c3c", "normal": "#2980b9"}
        colors  = [palette["anoxic"] if v else palette["normal"] for v in df[TARGET]]

        # ── Fig 1: Histogram of DO ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["DO"], bins=15, color="#2980b9", edgecolor="white", linewidth=0.8)
        ax.axvline(ANOXIC_THRESH, color="#e74c3c", linestyle="--", linewidth=1.8,
                   label=f"Anoxic Threshold ({ANOXIC_THRESH} mg/L)")
        ax.set_xlabel("Dissolved Oxygen (mg/L)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Dissolved Oxygen Levels\n(Station 1 | June 2022)", fontsize=13)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig1_do_histogram.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig1_do_histogram.png")

        # ── Fig 2: Scatter — Temp vs DO (anoxic color-coded) ───────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(df["TEMP"], df["DO"], c=df[TARGET],
                        cmap=LinearSegmentedColormap.from_list("aq", ["#2980b9", "#e74c3c"]),
                        edgecolors="white", linewidth=0.4, s=70, alpha=0.85)
        ax.axhline(ANOXIC_THRESH, color="#e74c3c", linestyle="--", linewidth=1.5,
                   label=f"Anoxic Threshold ({ANOXIC_THRESH} mg/L)")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Anoxic (1=Yes, 0=No)")
        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=12)
        ax.set_title("Temperature vs Dissolved Oxygen\n(Anoxic Conditions Highlighted)", fontsize=13)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig2_scatter_temp_do.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig2_scatter_temp_do.png")

        # ── Fig 3: Boxplot of key parameters ───────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        bp_data  = [df["DO"].values, df["TEMP"].values,
                    df["PH"].values, df["AMMONIA(mg/l)"].values]
        bp_labels = ["DO (mg/L)", "TEMP (°C)", "pH", "Ammonia (mg/L)"]
        bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        box_colors = ["#2980b9", "#e74c3c", "#27ae60", "#f39c12"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title("Boxplot of Key Water Quality Parameters\n(Station 1 | June 2022)", fontsize=13)
        ax.set_ylabel("Values", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig3_boxplot_parameters.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig3_boxplot_parameters.png")

        # ── Fig 4: Correlation Heatmap ──────────────────────────────────────
        cols = ["DO", "TEMP", "PH", "AMMONIA(mg/l)", "NITRATE(PPM)", "TURBIDITY"]
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Pearson r")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(["DO", "TEMP", "pH", "NH₃", "NO₃", "Turbidity"],
                            rotation=45, ha="right")
        ax.set_yticklabels(["DO", "TEMP", "pH", "NH₃", "NO₃", "Turbidity"])
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center",
                        va="center", fontsize=9,
                        color="white" if abs(corr.values[i,j]) > 0.5 else "black")
        ax.set_title("Pearson Correlation Heatmap\n(Water Quality Parameters)", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig4_correlation_heatmap.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig4_correlation_heatmap.png")

        # ── Fig 5: Anoxic vs Normal class count bar ─────────────────────────
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[TARGET].value_counts().sort_index()
        bars = ax.bar(["Normal (DO≥5.8)", "Anoxic (DO<5.8)"],
                       counts.values,
                       color=["#2980b9", "#e74c3c"], edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontweight="bold")
        ax.set_title("Class Distribution: Low vs Normal Dissolved Oxygen Conditions", fontsize=13)
        ax.set_ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig5_class_distribution.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig5_class_distribution.png")

    except Exception as e:
        raise RuntimeError(f"Static visualization failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — ANIMATION 1: DO Levels Over Time (chronological)
# ─────────────────────────────────────────────────────────────────────────────
def animate_do_over_time(df: pd.DataFrame):
    """Animated line chart showing DO evolution across June 2022."""
    try:
        df_sorted = df.sort_values("Date").reset_index(drop=True)
        x_vals = np.arange(len(df_sorted))
        y_vals = df_sorted["DO"].values
        colors = ["#e74c3c" if v else "#2980b9" for v in df_sorted[TARGET]]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(0, len(x_vals) - 1)
        ax.set_ylim(y_vals.min() - 0.3, y_vals.max() + 0.3)
        ax.axhline(ANOXIC_THRESH, color="#e74c3c", linestyle="--", linewidth=1.5,
                   label=f"Anoxic Threshold ({ANOXIC_THRESH} mg/L)", alpha=0.7)
        ax.set_xlabel("Sample Index (Sorted by Date)", fontsize=11)
        ax.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=11)
        ax.set_title("Animation 1: DO Level Progression — Station 1 | June 2022", fontsize=12)
        ax.legend(loc="upper right")

        line, = ax.plot([], [], color="#2980b9", linewidth=1.8, alpha=0.8)
        scat  = ax.scatter([], [], c=[], cmap=LinearSegmentedColormap.from_list(
            "aq", ["#2980b9","#e74c3c"]), s=45, zorder=5, vmin=0, vmax=1)
        timestamp = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10)

        xs_hist, ys_hist, cs_hist = [], [], []

        def update(frame):
            xs_hist.append(x_vals[frame])
            ys_hist.append(y_vals[frame])
            cs_hist.append(df_sorted[TARGET].iloc[frame])
            line.set_data(xs_hist, ys_hist)
            scat.set_offsets(np.column_stack([xs_hist, ys_hist]))
            scat.set_array(np.array(cs_hist))
            timestamp.set_text(f"Date: {df_sorted['Date'].iloc[frame].strftime('%b %d')}")
            return line, scat, timestamp

        ani = animation.FuncAnimation(fig, update, frames=len(x_vals),
                                      interval=120, blit=True, repeat=False)
        ani.save(f"{OUTPUT_DIR}/anim1_do_over_time.gif", writer="pillow", fps=8, dpi=120)
        plt.close()
        print("[ANIM] Saved anim1_do_over_time.gif")

    except Exception as e:
        raise RuntimeError(f"Animation 1 failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — ANIMATION 2: Rolling distribution shift of DO
# ─────────────────────────────────────────────────────────────────────────────
def animate_rolling_distribution(df: pd.DataFrame):
    """
    Animation 2: Rolling window histogram of DO showing distribution
    shift as we move through June 2022 chronologically.
    """
    try:
        df_sorted = df.sort_values("Date").reset_index(drop=True)
        do_vals   = df_sorted["DO"].values
        WINDOW    = 15
        n_frames  = len(do_vals) - WINDOW + 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Animation 2: Rolling DO Distribution Shift — Station 1 | June 2022",
                     fontsize=12, fontweight="bold")

        ax1.set_title("Rolling Window Histogram (n=15)", fontsize=11)
        ax1.set_xlabel("Dissolved Oxygen (mg/L)"); ax1.set_ylabel("Frequency")
        ax1.set_xlim(4.8, 7.0); ax1.set_ylim(0, 9)
        ax1.axvline(ANOXIC_THRESH, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8)

        ax2.set_title("Rolling Mean & Std of DO", fontsize=11)
        ax2.set_xlabel("End-of-Window Sample Index"); ax2.set_ylabel("DO (mg/L)")
        ax2.set_xlim(WINDOW-1, len(do_vals))
        ax2.set_ylim(do_vals.min() - 0.5, do_vals.max() + 0.5)
        ax2.axhline(ANOXIC_THRESH, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7)

        hist_bars  = ax1.bar([], [], width=0.15, color="#2980b9", edgecolor="white", alpha=0.85)
        mean_line, = ax2.plot([], [], color="#2980b9", linewidth=2, label="Rolling Mean")
        fill       = ax2.fill_between([], [], [], color="#2980b9", alpha=0.15, label="±1 Std")
        window_txt = ax1.text(0.02, 0.93, "", transform=ax1.transAxes, fontsize=9)
        ax2.legend(fontsize=9)

        mean_xs, mean_ys, std_ys = [], [], []

        # pre-build bin edges
        bin_edges = np.linspace(4.8, 7.0, 13)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        def update(frame):
            nonlocal fill
            window_data = do_vals[frame: frame + WINDOW]
            counts, _ = np.histogram(window_data, bins=bin_edges)

            # update histogram bars
            for rect, h in zip(ax1.patches, counts):
                rect.set_height(h)
            window_txt.set_text(
                f"Window: samples {frame}–{frame+WINDOW-1}\n"
                f"Mean DO: {np.mean(window_data):.3f} mg/L")

            end_idx = frame + WINDOW - 1
            mean_xs.append(end_idx)
            mean_ys.append(np.mean(window_data))
            std_ys.append(np.std(window_data))
            mean_line.set_data(mean_xs, mean_ys)

            # redraw fill_between
            for coll in ax2.collections:
                coll.remove()
            ax2.fill_between(mean_xs,
                              np.array(mean_ys) - np.array(std_ys),
                              np.array(mean_ys) + np.array(std_ys),
                              color="#2980b9", alpha=0.15)
            return ()

        # init histogram bars
        init_counts, _ = np.histogram(do_vals[:WINDOW], bins=bin_edges)
        ax1.bar(bin_centers, init_counts, width=(bin_edges[1]-bin_edges[0])*0.85,
                color="#2980b9", edgecolor="white", alpha=0.85)

        ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                      interval=200, blit=False, repeat=False)
        ani.save(f"{OUTPUT_DIR}/anim2_rolling_distribution.gif", writer="pillow", fps=5, dpi=120)
        plt.close()
        print("[ANIM] Saved anim2_rolling_distribution.gif")

    except Exception as e:
        raise RuntimeError(f"Animation 2 failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — MACHINE LEARNING: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train Logistic Regression, evaluate, and save confusion matrix plot."""
    try:
        X = df[FEATURES].to_numpy()
        y = df[TARGET].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

        from sklearn.model_selection import cross_val_score
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                        target_names=["Normal (0)", "Anoxic (1)"],
                                        output_dict=True)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)

        print(f"\n[ML] Test Accuracy: {acc:.4f}")
        print(f"[ML] 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("[ML] Confusion Matrix:")
        print(cm)
        print("[ML] Classification Report:")
        print(classification_report(y_test, y_pred,
                                     target_names=["Normal (0)", "Anoxic (1)"]))

        # Save confusion matrix figure
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=["Normal", "Anoxic"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix — Logistic Regression\n(Anoxic Event Classification)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig6_confusion_matrix.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig6_confusion_matrix.png")

        # Feature importance (coefficients)
        coef = np.abs(model.coef_[0])
        feat_importance = dict(zip(FEATURES, coef))
        sorted_feats = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        feat_names = [f[0].replace("(mg/l)","").replace("(PPM)","") for f, _ in sorted_feats]
        feat_vals  = [v for _, v in sorted_feats]
        ax.barh(feat_names[::-1], feat_vals[::-1], color="#2980b9", edgecolor="white")
        ax.set_xlabel("|Logistic Regression Coefficient|", fontsize=11)
        ax.set_title("Feature Importance\n(Anoxic Event Prediction Model)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig7_feature_importance.png", dpi=180)
        plt.close()
        print("[PLOT] Saved fig7_feature_importance.png")

        return {
            "accuracy": acc, "confusion_matrix": cm,
            "report": report, "model": model,
            "feature_importance": feat_importance,
            "cv_scores": cv_scores
        }

    except Exception as e:
        raise RuntimeError(f"ML training/evaluation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  AQUACULTURE WATER QUALITY ANALYSIS — AQU-01")
    print("  Anoxic Event Prediction | Station 1 | June 2022")
    print("="*65 + "\n")

    df_raw   = ingest_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_feat  = engineer_features(df_clean, threshold=ANOXIC_THRESH)
    stats    = statistical_analysis(df_feat)

    print("\n[PIPELINE] Generating static visualizations...")
    plot_static(df_feat)

    print("\n[PIPELINE] Generating animations...")
    animate_do_over_time(df_feat)
    animate_rolling_distribution(df_feat)

    print("\n[PIPELINE] Training and evaluating ML model...")
    ml_results = train_and_evaluate(df_feat)

    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE | Test Acc: {ml_results['accuracy']*100:.2f}%  |  "
          f"CV Acc: {ml_results['cv_scores'].mean()*100:.2f}%")
    print(f"  All outputs saved to: ./{OUTPUT_DIR}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

# ── NEW MODELS & COMPARISON FUNCTION

# ── MODULE 1: DATA INGESTION
def ingest_data(path: str) -> pd.DataFrame:
    """Load CSV dataset with error handling and initial inspection."""
    try:
        df = pd.read_csv(path)
        print(f"[INGESTION] Loaded {len(df)} records | Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Data ingestion failed: {e}")


# ── MODULE 2: DATA CLEANING & PREPROCESSING
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, enforce correct data types."""
    try:
        initial_len = len(df)
        df = df.drop_duplicates()
        print(f"[CLEANING] Removed {initial_len - len(df)} duplicate records.")
        
        numeric_cols = ["NITRATE(PPM)", "PH", "AMMONIA(mg/l)", "TEMP", "DO",
                        "TURBIDITY", "MANGANESE(mg/l)"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        missing_before = df.isnull().sum().sum()
        df = df.dropna(subset=numeric_cols)
        print(f"[CLEANING] Dropped {missing_before} missing values. Records remaining: {len(df)}")
        
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}")


# ── MODULE 3: FEATURE ENGINEERING & TARGET CREATION
def engineer_features(df: pd.DataFrame, threshold: float = ANOXIC_THRESH) -> pd.DataFrame:
    """Create binary anoxic label and validate features."""
    try:
        df[TARGET] = (df["DO"] < threshold).astype(int)
        df["day_of_month"] = df["Date"].dt.day
        print(f"[FEATURES] Anoxic threshold: DO < {threshold} mg/L")
        print(f"[FEATURES] Anoxic=1: {df[TARGET].sum()} | Normal=0: {(df[TARGET] == 0).sum()}")
        return df
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}")


# ── MODULE 4: ADVANCED FEATURE ENGINEERING
def create_polynomial_features(X: np.ndarray, feature_names: list, degree: int = 2) -> tuple:
    """Create polynomial features and interaction terms."""
    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        poly_names = poly.get_feature_names_out(feature_names)
        print(f"[POLY] Original: {X.shape[1]} features -> Polynomial: {X_poly.shape[1]} features")
        return X_poly, poly_names, poly
    except Exception as e:
        raise RuntimeError(f"Polynomial feature creation failed: {e}")


# ── MODULE 5: MULTI-MODEL COMPARISON WITH CROSS-VALIDATION
def run_model_comparison(X_dict: dict, y: np.ndarray, test_size: float = 0.2, 
                         random_state: int = 42) -> pd.DataFrame:
    """
    Compare multiple ML models with cross-validation.
    X_dict: dict of feature sets {'name': X_array}
    """
    results = []
    
    models = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=2000, 
                                                   class_weight='balanced', random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, 
                                                 class_weight='balanced', random_state=random_state),
        'SVM (RBF)': SVC(C=1.0, kernel='rbf', class_weight='balanced', 
                         probability=True, random_state=random_state),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    
    for feat_name, X in X_dict.items():
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n{'='*55}")
        print(f"FEATURE SET: {feat_name} (n_features={X.shape[1]})")
        print('='*55)
        
        for model_name, model in models.items():
            start = time.perf_counter()
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            test_acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            elapsed = time.perf_counter() - start
            
            result = {
                'Feature_Set': feat_name,
                'Model': model_name,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'Test_Acc': test_acc,
                'AUC_ROC': auc,
                'Time_sec': elapsed
            }
            results.append(result)
            
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            print(f"{model_name:20s} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | "
                  f"Test: {test_acc:.3f} | AUC: {auc_str} | {elapsed:.3f}s")
    
    return pd.DataFrame(results)


# ── MODULE 6: HYPERPARAMETER TUNING (GridSearchCV)
def tune_model(model, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray,
               cv=None, scoring: str = 'accuracy') -> dict:
    """Perform GridSearchCV hyperparameter tuning."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    start = time.perf_counter()
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=1)    
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'best_estimator': grid.best_estimator_,
        'cv_results': pd.DataFrame(grid.cv_results_),
        'time_sec': elapsed
    }


# ── VISUALIZATION: Model Comparison Bar Chart
def plot_model_comparison(results_df: pd.DataFrame, output_dir: str):
    """Generate bar chart comparing CV and Test accuracy across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models_list = ['Logistic\nRegression', 'Random\nForest', 'SVM\n(RBF)', 'KNN\n(k=5)']
    
    # Extract data for original features
    orig = results_df[results_df['Feature_Set'] == 'Original (5 features)']
    poly = results_df[results_df['Feature_Set'] == 'Polynomial (20 features)']
    
    orig_cv = orig['CV_Mean'].values
    orig_test = orig['Test_Acc'].values
    poly_cv = poly['CV_Mean'].values
    poly_test = poly['Test_Acc'].values
    
    x = np.arange(len(models_list))
    width = 0.35
    
    # Plot 1: CV Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, orig_cv, width, label='Original (5 feat)', 
                    color='#2980b9', alpha=0.85, edgecolor='white')
    bars2 = ax1.bar(x + width/2, poly_cv, width, label='Polynomial (20 feat)', 
                    color='#e74c3c', alpha=0.85, edgecolor='white')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess (50%)')
    ax1.set_ylabel('5-Fold CV Accuracy', fontsize=11)
    ax1.set_title('Cross-Validation Accuracy Comparison\n(Anoxic Event Prediction)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, fontsize=10)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, 0.7)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Test Accuracy
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, orig_test, width, label='Original (5 feat)', 
                    color='#2980b9', alpha=0.85, edgecolor='white')
    bars4 = ax2.bar(x + width/2, poly_test, width, label='Polynomial (20 feat)', 
                    color='#e74c3c', alpha=0.85, edgecolor='white')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess (50%)')
    ax2.set_ylabel('Test Set Accuracy', fontsize=11)
    ax2.set_title('Test Accuracy Comparison\n(Anoxic Event Prediction)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_list, fontsize=10)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0, 0.7)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_model_comparison.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved fig_model_comparison.png")


# ── VISUALIZATION: Feature Importance Across All Models
def plot_feature_importance_all_models(X_orig: np.ndarray, y: np.ndarray, 
                                       feature_names: list, output_dir: str):
    """Generate feature importance plots for all four models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model 1: Logistic Regression
    ax1 = axes[0, 0]
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    coefs = np.abs(lr.coef_[0])
    sorted_idx = np.argsort(coefs)[::-1]
    colors = ['#e74c3c' if c > 0.3 else '#2980b9' if c > 0.15 else '#95a5a6' 
              for c in coefs[sorted_idx]]
    ax1.barh(range(len(feature_names)), coefs[sorted_idx], color=colors, 
             edgecolor='white', alpha=0.85)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax1.set_xlabel('|Coefficient|', fontsize=11)
    ax1.set_title('Logistic Regression\nFeature Importance', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    for i, v in enumerate(coefs[sorted_idx]):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Model 2: Random Forest
    ax2 = axes[0, 1]
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    colors = ['#e74c3c' if i > 0.25 else '#2980b9' if i > 0.15 else '#95a5a6' 
              for i in importances[sorted_idx]]
    ax2.barh(range(len(feature_names)), importances[sorted_idx], color=colors, 
             edgecolor='white', alpha=0.85)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax2.set_xlabel('Gini Importance', fontsize=11)
    ax2.set_title('Random Forest\nFeature Importance', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    for i, v in enumerate(importances[sorted_idx]):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Model 3: SVM (Linear kernel for interpretability)
    ax3 = axes[1, 0]
    svm_linear = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)
    svm_linear.fit(X_train, y_train)
    svm_coefs = np.abs(svm_linear.coef_[0])
    sorted_idx = np.argsort(svm_coefs)[::-1]
    colors = ['#e74c3c' if c > 0.3 else '#2980b9' if c > 0.15 else '#95a5a6' 
              for c in svm_coefs[sorted_idx]]
    ax3.barh(range(len(feature_names)), svm_coefs[sorted_idx], color=colors, 
             edgecolor='white', alpha=0.85)
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax3.set_xlabel('|Linear SVM Weight|', fontsize=11)
    ax3.set_title('SVM (Linear Kernel)\nFeature Importance', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    for i, v in enumerate(svm_coefs[sorted_idx]):
        ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Model 4: KNN (proxy via feature-target correlation)
    ax4 = axes[1, 1]
    correlations = [np.corrcoef(X_orig[:, i], y)[0, 1] for i in range(len(feature_names))]
    abs_corr = np.abs(correlations)
    sorted_idx = np.argsort(abs_corr)[::-1]
    colors = ['#e74c3c' if c > 0.15 else '#2980b9' if c > 0.08 else '#95a5a6' 
              for c in abs_corr[sorted_idx]]
    ax4.barh(range(len(feature_names)), abs_corr[sorted_idx], color=colors, 
             edgecolor='white', alpha=0.85)
    ax4.set_yticks(range(len(feature_names)))
    ax4.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax4.set_xlabel('|Pearson r with Target|', fontsize=11)
    ax4.set_title('KNN Proxy\n(Feature-Target Correlation)', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    for i, v in enumerate(abs_corr[sorted_idx]):
        ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    fig.suptitle('Feature Importance Across Four ML Models\n(Anoxic Event Prediction — Original Features)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_feature_importance_all_models.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved fig_feature_importance_all_models.png")


# ── VISUALIZATION: Confusion Matrices for All Models
def plot_confusion_matrices_all_models(X_orig: np.ndarray, y: np.ndarray, output_dir: str):
    """Generate confusion matrices for all four models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_cm = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=2000, 
                                                   class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, 
                                                 class_weight='balanced', random_state=42),
        'SVM (RBF)': SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }
    
    positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for (name, model), (row, col) in zip(models_cm.items(), positions):
        ax = axes[row, col]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Plot
        im = ax.imshow(cm, cmap='Blues', alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anoxic'], fontsize=10)
        ax.set_yticklabels(['Normal', 'Anoxic'], fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(f'{name}\nAcc={acc:.3f} | Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}', 
                     fontsize=11, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i,j] > cm.max()/2 else "black", 
                              fontsize=14, fontweight='bold')
    
    fig.suptitle('Confusion Matrices — Four ML Models\n(Anoxic Event Prediction, Original Features)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_confusion_matrices_all_models.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved fig_confusion_matrices_all_models.png")


# ── VISUALIZATION: Cross-Validation Stability Boxplot
def plot_cv_stability(X_orig: np.ndarray, y: np.ndarray, output_dir: str):
    """Generate boxplot showing CV score distributions across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_cv = {
        'Logistic\nRegression': LogisticRegression(C=1.0, max_iter=2000, 
                                                    class_weight='balanced', random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=100, max_depth=5, 
                                                  class_weight='balanced', random_state=42),
        'SVM\n(RBF)': SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42),
        'KNN\n(k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }
    
    cv_data = []
    labels = []
    for name, model in models_cv.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_data.append(scores)
        labels.append(name)
        print(f"{name.replace(chr(10), ' ')}: {scores.mean():.4f} ± {scores.std():.4f} | "
              f"Range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    bp = ax.boxplot(cv_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    widths=0.6)
    
    colors = ['#2980b9', '#27ae60', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor('white')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guess (50%)')
    ax.set_ylabel('5-Fold CV Accuracy', fontsize=12)
    ax.set_title('Cross-Validation Stability Across Four ML Models\n'
                 '(Anoxic Event Prediction — Station 1, June 2022)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.2, 0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')
    
    for i, scores in enumerate(cv_data):
        ax.scatter(i+1, scores.mean(), color='white', s=100, zorder=5, 
                  edgecolor='black', linewidth=1.5, marker='D')
        ax.text(i+1, scores.mean()+0.03, f'μ={scores.mean():.3f}', 
               ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_cv_stability_boxplot.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("[PLOT] Saved fig_cv_stability_boxplot.png")


# ── MAIN EXECUTION
if __name__ == "__main__":
    # Data pipeline
    df_raw = ingest_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean, threshold=ANOXIC_THRESH)
    
    # Prepare features
    y = df_feat[TARGET].values
    X_orig = df_feat[FEATURES].values
    
    # Polynomial features
    X_poly, poly_names, _ = create_polynomial_features(X_orig, FEATURES, degree=2)
    
    # Model comparison
    X_dict = {
        'Original (5 features)': X_orig,
        'Polynomial (20 features)': X_poly
    }
    
    results_df = run_model_comparison(X_dict, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_results.csv", index=False)
    print("\n[CSV] Saved model_comparison_results.csv")
    
    # Generate all visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_model_comparison(results_df, OUTPUT_DIR)
    plot_feature_importance_all_models(X_orig, y, FEATURES, OUTPUT_DIR)
    plot_confusion_matrices_all_models(X_orig, y, OUTPUT_DIR)
    plot_cv_stability(X_orig, y, OUTPUT_DIR)
    
    # Hyperparameter tuning example: KNN
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: KNN")
    print("="*60)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    tuned_knn = tune_model(KNeighborsClassifier(), param_grid_knn, X_train, y_train)
    print(f"Best Params: {tuned_knn['best_params']}")
    print(f"Best CV Score: {tuned_knn['best_score']:.4f}")
    print(f"Grid Search Time: {tuned_knn['time_sec']:.3f}s")
    
    # Evaluate tuned model
    best_knn = tuned_knn['best_estimator']
    y_pred_knn = best_knn.predict(X_test)
    print(f"Test Accuracy (Tuned KNN): {accuracy_score(y_test, y_pred_knn):.4f}")
    
    # Hyperparameter tuning: Random Forest
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: Random Forest")
    print("="*60)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5]
    }
    
    tuned_rf = tune_model(
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
        param_grid_rf, X_train, y_train
    )
    print(f"Best Params: {tuned_rf['best_params']}")
    print(f"Best CV Score: {tuned_rf['best_score']:.4f}")
    print(f"Grid Search Time: {tuned_rf['time_sec']:.3f}s")
    
    best_rf = tuned_rf['best_estimator']
    y_pred_rf = best_rf.predict(X_test)
    print(f"Test Accuracy (Tuned RF): {accuracy_score(y_test, y_pred_rf):.4f}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)