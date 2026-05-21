# COMPROG1 Final Project
## Automated Analysis of Aquaculture Water Quality Stability and Evaluation of Predictive Modeling Using Python Data Pipelines

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-Academic-orange)](./LICENSE)

> **A Python-based data pipeline for analyzing aquaculture water quality and evaluating predictive machine learning models under stable environmental conditions.**

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Citation](#citation)
- [Author](#author)
- [Acknowledgment](#acknowledgment)
- [AI Disclosure](#ai-disclosure)

---

## Overview

This project presents an automated Python-based data pipeline for analyzing aquaculture water quality and evaluating the performance of predictive machine learning models under stable environmental conditions.

The study focuses on **Station 1 data during June 2022**, a high-temperature period associated with reduced oxygen solubility. The workflow includes data ingestion, cleaning, feature selection, feature engineering, statistical analysis, and multi-model evaluation.

**Main objectives:**

1. To analyze the stability of dissolved oxygen (DO) levels under high-temperature conditions using an automated statistical pipeline
2. To evaluate the effectiveness of multiple machine learning models under stable environmental conditions

The study found that although temperature was elevated, dissolved oxygen remained relatively stable, and machine learning models showed consistently low predictive performance. This indicates that **limited dataset variability and weak feature-target relationships constrained model effectiveness**.

---

## Dataset

The study used aquaculture water quality data filtered through a targeted selection process.

**Filtering scope:**

| Field | Value |
|:------|:------|
| Station | Station 1 |
| Period | June 2022 |

**Water quality parameters used:**

| Parameter | Unit | Role |
|:----------|:-----|:-----|
| Temperature | °C | Feature |
| Dissolved Oxygen (DO) | mg/L | Target / Feature |
| pH | — | Feature |
| Ammonia | mg/L | Feature |
| Nitrate | mg/L | Feature |
| Turbidity | NTU | Feature |

The dataset was processed after filtering for the selected station and time period, followed by data cleaning procedures such as removing missing values and duplicate entries.

---

## Methodology

The project follows a structured data pipeline consisting of the following stages:

    RAW DATA
        │
        ▼
    [1] Data Ingestion ──────────── Load CSV via Pandas
        │
        ▼
    [2] Initial Inspection ─────── Understand structure and variables
        │
        ▼
    [3] Targeted Filtering ─────── Station 1 | June 2022
        │
        ▼
    [4] Data Preprocessing ─────── Drop nulls and duplicates
        │
        ▼
    [5] Feature Selection ──────── Temp, DO, pH, Ammonia, Nitrate, Turbidity
        │
        ▼
    [6] Feature Engineering ────── Polynomial expansion (degree = 2)
        │
        ▼
    [7] Exploratory Analysis ────── Stats, correlations, visualizations
        │
        ▼
    [8] Model Training ─────────── 80/20 split | 5-fold stratified CV
        │          └── Logistic Regression
        │          └── Random Forest
        │          └── Support Vector Machine
        │          └── K-Nearest Neighbors
        ▼
    [9] Hyperparameter Tuning ───── GridSearchCV
        │
        ▼
    [10] Performance Evaluation ─── Accuracy | Precision | Recall
                                     F1-score | AUC-ROC | Confusion Matrix

---

## Results

### Environmental Stability Findings

| Metric | Value |
|:-------|:------|
| Mean Dissolved Oxygen (DO) | 5.77 mg/L |
| DO Standard Deviation | 0.42 |
| Mean Temperature | 27.92°C |
| Temperature-DO Correlation | r = -0.13 |

These findings indicate that dissolved oxygen remained **moderate and relatively stable** despite elevated temperature. The inverse relationship between temperature and dissolved oxygen was weak within the filtered dataset.

---

### Predictive Modeling Results

| Model | Feature Set | CV Accuracy | Test Accuracy | AUC-ROC |
|:------|:-----------|:------------|:-------------|:--------|
| KNN (k=5) | Original (5 features) | 45.8% | 38.5% | 0.333 |
| KNN (k=5) | Polynomial (20 features) | 47.3% | 46.2% | 0.381 |
| **KNN (Tuned)** | **Best config** | **58.4%** | **38.5%** | **—** |

> **Note:** All test accuracies remain below the 50% random-guess baseline. Polynomial expansion and GridSearchCV failed to improve performance, confirming **data variability** — not model inadequacy — as the fundamental constraint.

---

### Hyperparameter Tuning Details

**KNN Grid — 28 combinations:**

| Parameter | Values |
|:----------|:-------|
| n_neighbors | 3, 5, 7, 9, 11, 13, 15 |
| weights | uniform, distance |
| metric | euclidean, manhattan |

**Random Forest Grid — 24 combinations:**

| Parameter | Values |
|:----------|:-------|
| n_estimators | 50, 100, 200 |
| max_depth | 3, 5, 7, None |
| min_samples_split | 2, 5 |

**Best KNN Configuration:** `k=11`, `weights='distance'`, `metric='manhattan'`

---

### Conclusion

The study concludes that **stable environmental conditions with limited variability reduce the effectiveness of predictive machine learning models**. The project demonstrates that automated data pipelines are useful not only for environmental monitoring but also for identifying dataset limitations and assessing whether machine learning models are suitable for a given dataset.

---

## Repository Structure

    COMPROG1_FINAL-PROJECT/
    │
    ├── 📓 Machine_Learning_Code.ipynb     # Main analysis notebook
    ├── 🐍 main.py                         # Standalone pipeline script
    ├── 📊 filtered_dataset.csv            # Preprocessed and filtered dataset
    ├── 📦 requirements.txt                # Python dependencies
    ├── 📄 CRESENCIO_2043_IEEE_Paper.pdf   # Full IEEE-format research paper
    │
    └── 📁 outputs/
        │
        ├── 📈 Figures
        │   ├── fig_model_comparison.png           # Side-by-side model metric comparison
        │   ├── fig_model_comparison_vertical.png  # Vertical layout model comparison
        │   ├── fig_confusion_matrices.png         # Confusion matrices for all models
        │   ├── fig_feature_importance.png         # Random Forest feature importances
        │   ├── fig_cv_stability_boxplot.png       # Cross-validation score distributions
        │   ├── fig_correlation_heatmap.png        # Feature correlation heatmap
        │   ├── fig_class_distribution.png         # Target class distribution
        │   ├── fig_histogram_do.png               # DO level frequency histogram
        │   ├── fig_scatter_temp_do.png            # Temperature vs DO scatter plot
        │   └── fig_boxplot_parameters.png         # Boxplots for all water parameters
        │
        ├── 🎞️ Animations
        │   ├── anim1_do_progression.gif           # DO level changes over time
        │   └── anim2_rolling_distribution.gif     # Rolling distribution animation
        │
        └── 📋 Reports
            ├── complete_model_results.csv         # Full results across all configurations
            └── model_comparison_results.csv       # Summarized model comparison table

---

## Usage

### Setup

Clone the repository and install dependencies:

    git clone https://github.com/centaurids-hub/COMPROG1_FINAL-PROJECT.git
    cd COMPROG1_FINAL-PROJECT
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

On Windows, activate with:

    venv\Scripts\activate

### Run

Launch the notebook:

    jupyter notebook Machine_Learning_Code.ipynb

Or run the pipeline script directly:

    python main.py

---

## Tech Stack

| Category | Tools |
|:---------|:------|
| Language | Python 3.10+ |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Animation | matplotlib.animation |
| Environment | Jupyter Notebook, Python script |

---

## Citation

If referencing this work:

> Cresencio, J. S. (2026). *Automated Analysis of Aquaculture Water Quality Stability and Evaluation of Predictive Modeling Using Python Data Pipelines*. COMPROG1 Final Project, Technological University of the Philippines — Manila.

Full paper: [CRESENCIO_2043_IEEE_Paper.pdf](./CRESENCIO_2043_IEEE_Paper.pdf)

---

## Author

**Jesier S. Cresencio**
Department of Electronics Engineering
Technological University of the Philippines, Manila
jesiercresencio12@gmail.com

---

## Acknowledgment

### Human Contribution Disclosure

Portions of this project also benefited from the technical guidance of the author's friend's brother, a **Senior Web Developer at IBM**, whose professional expertise in software engineering, systems architecture, and full-stack development contributed to the following areas:

- Code structure review and software engineering best practices
- Guidance on modular design, version control workflows, and GitHub repository organization
- Advice on production-level Python practices and pipeline optimization
- Technical feedback on data pipeline architecture and project documentation standards

His industry experience in enterprise-level software development helped improve the engineering discipline and professionalism reflected in this project's codebase and documentation.

All programming logic, data analysis, statistical interpretation, and written content remain the original work of the author.

---

## AI Disclosure

This project utilized AI tools as supplementary assistants during the development, debugging, documentation, formatting, and refinement stages of the study. The following AI systems were used:

- ChatGPT — https://chatgpt.com
- Claude — https://claude.ai
- Grok — https://grok.com
- Kimi AI — https://kimi.ai

These tools were primarily used for:

- Code debugging and syntax assistance
- Documentation refinement and proofreading
- Explanation of machine learning concepts and Python libraries
- Suggestions for formatting and project organization
- General programming guidance and workflow optimization

All final decisions, implementations, data processing, analysis, interpretation of results, and conclusions were independently reviewed, validated, and finalized by the author. AI-generated suggestions were treated only as assistive references and not as substitutes for critical analysis, programming logic, or academic judgment.

The author remains fully responsible for the accuracy, originality, integrity, and overall content of this project.

---

<div align="center">

**Developed under the Technological University of the Philippines — Manila.**

Dataset sourced from [Kaggle](https://www.kaggle.com/datasets)

</div>
````