# COMPROG1 Final Project
## Automated Analysis of Aquaculture Water Quality Stability and Evaluation of Predictive Modeling Using Python Data Pipelines

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green)](https://scikit-learn.org)

**A Python-based data pipeline for analyzing aquaculture water quality and evaluating predictive modeling performance under stable environmental conditions.**

[Overview](#overview) • [Dataset](#dataset) • [Methodology](#methodology) • [Results](#results) • [Repository Reference](#repository-reference) • [Citation](#citation)

---
## Overview

This project presents an automated Python-based data pipeline for analyzing aquaculture water quality and evaluating the performance of predictive machine learning models under stable environmental conditions.

The study focuses on **Station 1 data during June 2022**, a high-temperature period associated with reduced oxygen solubility. The workflow includes data ingestion, cleaning, feature selection, feature engineering, statistical analysis, and multi-model evaluation.

The main objective of the study is twofold:

1. To analyze the stability of dissolved oxygen (DO) levels under high-temperature conditions using an automated statistical pipeline
2. To evaluate the effectiveness of multiple machine learning models under stable environmental conditions

The study found that although temperature was elevated, dissolved oxygen remained relatively stable, and machine learning models showed consistently low predictive performance. This indicates that **limited dataset variability and weak feature-target relationships constrained model effectiveness**.

---

## Dataset

Based on the paper, the study used aquaculture water quality data filtered through a targeted selection process.

**Filtering scope used in the study:**
- **Station 1**
- **June 2022**

**Water quality parameters used:**
- Temperature
- Dissolved Oxygen (DO)
- pH
- Ammonia
- Nitrate
- Turbidity

The paper states that the dataset was processed after filtering for the selected station and time period, followed by data cleaning procedures such as removing missing values and duplicate entries.

---

## Methodology

The project follows a structured data pipeline consisting of the following stages:

1. **Data Ingestion**  
   The dataset was loaded from a CSV file into the Python environment using Pandas.

2. **Initial Data Inspection**  
   Initial inspection was performed to understand the dataset structure and variables.

3. **Targeted Data Filtering**  
   The study isolated **Station 1 data during June 2022** to examine water quality under high-temperature conditions.

4. **Data Preprocessing**  
   Missing values and duplicate entries were removed to improve data quality and consistency.

5. **Feature Selection**  
   The selected variables included:
   - Temperature
   - Dissolved Oxygen
   - pH
   - Ammonia
   - Nitrate
   - Turbidity

6. **Feature Engineering**  
   Polynomial feature expansion with **degree = 2** was applied to generate interaction terms and squared features.

7. **Exploratory Data Analysis**  
   Statistical analysis and visualization were used to identify patterns and relationships among variables.

8. **Model Development and Evaluation**  
   The dataset was divided into **80/20 training and testing subsets** with stratification.  
   Four machine learning models were evaluated using **5-fold stratified cross-validation**:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
   - K-Nearest Neighbors

9. **Hyperparameter Tuning**  
   **GridSearchCV** was used to optimize model configurations.

10. **Performance Evaluation**  
    Models were evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - AUC-ROC
    - Confusion Matrix

---

## Results

### Environmental Stability Findings

The statistical analysis in the paper reported the following:

- **Mean Dissolved Oxygen (DO):** 5.77 mg/L  
- **DO Standard Deviation:** 0.42  
- **Mean Temperature:** 27.92°C  
- **Temperature-DO Correlation:** r = -0.13  

These findings indicate that dissolved oxygen remained **moderate and relatively stable** despite elevated temperature, and the inverse relationship between temperature and dissolved oxygen was weak within the filtered dataset.

### Predictive Modeling Results

The paper reports consistently low predictive performance across all tested machine learning models and configurations.

- **Highest cross-validation accuracy:** 58.4%  
- **Best-performing model:** Tuned K-Nearest Neighbors  
- **Reported test performance:** below 46.2%

The results indicate that increasing model complexity through polynomial expansion and hyperparameter tuning did not significantly improve performance.

### Conclusion

The study concludes that **stable environmental conditions with limited variability reduce the effectiveness of predictive machine learning models**. The project demonstrates that automated data pipelines are useful not only for environmental monitoring but also for identifying dataset limitations and assessing whether machine learning models are suitable for a given dataset.

---

## Repository Reference

This repository is intended for the final project submission and documentation aligned with the paper.

For the supporting implementation files and code documentation in notebook format, refer to the separate repository:

**Code Documentation Repository:**  
https://github.com/centaurids-hub/COMPROG1_FINAL-PROJECT

This referenced repository also contains the paper and related project files used for documentation purposes.

---

## Citation

If referencing this work:

> Crescencio, J. S. (2026). *Automated Analysis of Aquaculture Water Quality Stability and Evaluation of Predictive Modeling Using Python Data Pipelines*. COMPROG1 Final Project, Technological University of the Philippines – Manila.

---

## Author

**Jesier S. Cresencio**  
Department of Electronics Engineering  
Technological University of the Philippines, Manila  
Email: `jesiercresencio12@gmail.com`

---

## Acknowledgment

### Human Contribution Disclosure

Portions of this project also benefited from the technical guidance of the author’s friend’s brother, a **Senior Web Developer at IBM**, whose professional expertise in software engineering, systems architecture, and full-stack development contributed to the following areas:

- Code structure review and software engineering best practices  
- Guidance on modular design, version control workflows, and GitHub repository organization  
- Advice on production-level Python practices and pipeline optimization  
- Technical feedback on data pipeline architecture and project documentation standards  

His industry experience in enterprise-level software development helped improve the engineering discipline and professionalism reflected in this project’s codebase and documentation.

All programming logic, data analysis, statistical interpretation, and written content remain the original work of the author.

---

## AI Disclosure

### Artificial Intelligence (AI) Usage Disclosure

This project utilized Artificial Intelligence (AI) tools as supplementary assistants during the development, debugging, documentation, formatting, and refinement stages of the study. The following AI systems were used throughout the project:

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
```| KNN (k=5) | Original (5) | 45.8% | 38.5% | 0.333 |
| KNN (k=5) | Polynomial (20) | 47.3% | 46.2% | 0.381 |
| **KNN (Tuned)** | — | **58.4%** | **38.5%** | — |

> **Note:** All test accuracies remain below the 50% random-guess baseline. Polynomial expansion and GridSearchCV failed to improve performance, confirming **data variability** — not model inadequacy — as the fundamental constraint.

<details>
<summary><b>Hyperparameter Tuning Details</b></summary>

```
KNN Grid (28 combinations):
  n_neighbors: {3, 5, 7, 9, 11, 13, 15}
  weights: {uniform, distance}
  metric: {euclidean, manhattan}

Random Forest Grid (24 combinations):
  n_estimators: {50, 100, 200}
  max_depth: {3, 5, 7, None}
  min_samples_split: {2, 5}

Best KNN Config: k=11, weights='distance', metric='manhattan'
```

</details>

---

## Repository Structure

```
COMPROG1_FINAL-PROJECT/
|
|-- Machine_Learning_Code.ipynb      # Main analysis notebook
|-- main.py                           # Pipeline script
|-- filtered_dataset.csv              # Preprocessed dataset
|-- requirements.txt                  # Python dependencies
|-- CRESENCIO_2043_IEEE_Paper.pdf    # Full research paper
|
|-- outputs/                          # Generated figures & animations
|   |-- fig_model_comparison.png
|   |-- fig_model_comparison_vertical.png
|   |-- fig_confusion_matrices.png
|   |-- fig_feature_importance.png
|   |-- fig_cv_stability_boxplot.png
|   |-- fig_correlation_heatmap.png
|   |-- fig_class_distribution.png
|   |-- fig_histogram_do.png
|   |-- fig_scatter_temp_do.png
|   |-- fig_boxplot_parameters.png
|   |-- anim1_do_progression.gif
|   |-- anim2_rolling_distribution.gif
|   |
|   |-- complete_model_results.csv
|   |-- model_comparison_results.csv
```

---

## Usage

### Setup

```bash
# Clone the repository
git clone https://github.com/centaurids-hub/COMPROG1_FINAL-PROJECT.git
cd COMPROG1_FINAL-PROJECT

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Launch Jupyter Notebook
jupyter notebook Machine_Learning_Code.ipynb

# Or run the pipeline script
python main.py
```

---

## Tech Stack

| Category | Tools |
|:---------|:------|
| Language | `Python 3.10+` |
| Data Processing | `pandas`, `NumPy` |
| Machine Learning | `scikit-learn` |
| Visualization | `Matplotlib`, `Seaborn` |
| Animation | `matplotlib.animation` |
| Environment | `Python script` |

---

## Citation

If referencing this work in your research:

> Cresencio, J. S. (2026). *Automated Analysis of Aquaculture Water Quality Stability and Evaluation of Predictive Modeling Using Python Data Pipelines*. COMPROG1 Final Project, Technological University of the Philippines — Manila.

Full paper: [`CRESENCIO_2043_IEEE_Paper.pdf`](./CRESENCIO_2043_IEEE_Paper.pdf)

---

## Author

**Jesier S. Cresencio**  
Department of Electronics Engineering  
Technological University of the Philippines, Manila  
`jesiercresencio12@gmail.com`

---

<div align="center">

**Developed under the Technological University of the Philippines — Manila.**

Dataset sourced from <a href="https://www.kaggle.com/datasets">Kaggle</a>

</div>

## AI Disclosure

### Artificial Intelligence (AI) Usage Disclosure
This project utilized Artificial Intelligence (AI) tools as supplementary assistants during the development, debugging, documentation, formatting, and refinement stages of the study. The following AI systems were used throughout the project:

- ChatGPT — https://chatgpt.com
- Claude — https://claude.ai
- Grok — https://grok.com
- Kimi AI — https://kimi.ai

These tools were primarily used for:
- Code debugging and syntax assistance
- Documentation refinement and proofreading
- Explanation of machine learning concepts and Python libraries
- Suggestions for visualization formatting and project structure
- General programming guidance and workflow optimization

All final decisions, implementations, data processing, analysis, interpretation of results, and conclusions were independently reviewed, validated, and finalized by the author. AI-generated suggestions were treated only as assistive references and not as substitutes for critical analysis, programming logic, or academic judgment.

The author remains fully responsible for the accuracy, originality, integrity, and overall content of this project.
