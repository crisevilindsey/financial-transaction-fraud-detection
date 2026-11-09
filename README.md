# Predictive Analytics for Financial Transaction Fraud Detection

**Criselvi Lindsey** | Data Science · Machine Learning · Risk Analytics

An end-to-end machine-learning study of financial transaction fraud using the **BankSim synthetic bank payments dataset**. The project compares classification approaches, evaluates the trade-off between missed fraud and false alerts, and examines how model performance changes under different validation and feature-selection choices.

> **Portfolio highlight:** A tuned histogram-based gradient-boosting model, evaluated at a threshold selected using out-of-fold training predictions, identified **1,145 of 1,440 fraudulent transactions** in the held-out test set, generating **225 false alerts**. These are results on synthetic data, not estimates of real-world deployment performance.

## Business problem

Financial institutions need to identify suspicious transactions without overwhelming investigators or unnecessarily inconveniencing customers. Because fraud is rare, overall classification accuracy can conceal missed fraud. This analysis emphasizes fraud recall, alert precision, and the workload created by false positives.

The research addresses four questions:

1. How effectively can classification models distinguish fraudulent from legitimate transactions?
2. Which transaction characteristics are most useful for identifying fraud?
3. How do models compare in detecting fraud while minimizing false alerts?
4. How does class imbalance affect model performance?

## Dataset

The study uses **BankSim**, a synthetic dataset developed for fraud-detection research. The analyzed data include **594,643 transactions** over approximately six simulated months: **587,443 legitimate transactions** and **7,200 fraudulent transactions** (**1.21% fraud**).

Key variables include transaction sequence (`step`), coded age and gender, merchant, transaction category, transaction amount, and the binary fraud label. The study excludes the constant ZIP-code fields and the customer identifier from modeling.

**Data availability:** The CSV is **not included in this repository**. Obtain the BankSim dataset from its original research/distribution source, and place the file named `bs140513_032310.csv` in the notebook's working directory before running the analysis. Check the dataset's distribution terms before redistributing it.

## Methods

The analysis compares four classification approaches:

- Logistic regression — interpretable baseline
- Class-weighted logistic regression — tests a direct response to class imbalance
- Random forest — captures nonlinear relationships and interactions
- Histogram-based gradient boosting — an efficient boosted-tree approach for tabular data

The workflow includes data auditing and exploratory analysis, preprocessing within model pipelines, an **80/20 stratified train/test split**, **five-fold cross-validation**, hyperparameter search, decision-threshold selection using out-of-fold training predictions, feature ablation, time-based validation, and subgroup evaluation. The held-out test set contains **118,929 transactions**, including **1,440 fraud cases**.

Performance is assessed using precision, recall, F1, **precision–recall area under the curve (PR-AUC)**, false-positive and false-negative rates, and confusion-matrix counts.

## Results

### Four-model comparison at the default 0.50 threshold

| Model | Precision | Recall | F1 | PR-AUC | Fraud detected | False alerts |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.870 | 0.719 | 0.788 | 0.883 | 1,036 | 155 |
| Class-weighted logistic regression | 0.271 | 0.983 | 0.425 | 0.877 | 1,416 | 3,804 |
| Random forest | 0.851 | 0.759 | 0.803 | 0.872 | 1,093 | 191 |
| Gradient boosting | 0.864 | 0.770 | 0.815 | 0.896 | 1,109 | 174 |

*Source: final project white paper, Table 2. These are held-out test-set results on synthetic data.*

### Operational trade-off after tuning

| Tuned gradient-boosting operating point | Threshold | Recall | Precision | Fraud detected | Fraud missed | False alerts |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.753 | 0.873 | 1,084 | 356 | 157 |
| F1-selected | 0.36 | 0.795 | 0.836 | 1,145 | 295 | 225 |
| Recall-targeted | 0.22 | 0.847 | 0.768 | 1,220 | 220 | 368 |

*Source: final project white paper, Table 4. Thresholds were chosen from out-of-fold training predictions, then evaluated on the held-out test set. The recall-targeted threshold fell slightly short of 0.85 recall on the test set.*

**Interpretation:** Increasing sensitivity catches more fraud but creates more alerts for investigators to review. The appropriate threshold depends on investigation capacity, the cost of missed fraud, and customer impact; it cannot be chosen from F1 alone.

### Additional findings

- **Important predictors:** Transaction amount, category, and merchant were most influential in the fitted gradient-boosting model's permutation-importance analysis.
- **Merchant dependence:** Removing merchant reduced random-split PR-AUC from approximately **0.896 to 0.805**, raising concerns about transferability to different merchants or real banking environments.
- **Class imbalance:** Class-weighted logistic regression achieved high recall but produced substantially more false alerts.
- **Subgroup review:** Recall for the oldest coded age group was lower in the synthetic test data, but the overall age-group comparison was not statistically significant; the subgroup sample was small.

## Practical relevance

This project illustrates how a risk analytics team might evaluate a fraud-screening model as a **decision-support tool**. A production implementation would require representative institutional data, prospective or temporal testing, probability calibration, investigation-cost analysis, ongoing monitoring, and human review of consequential alerts.

## Repository files

- `fraud_detection_analysis.ipynb` — final analysis notebook, including preparation, modeling, validation, and visualizations (**add this file to the repository**).
- `fraud_detection_white_paper.pdf` — detailed research report, figures, interpretation, and ethical assessment (**add this file to the repository**).
- `README.md` — project overview and instructions (this document).

The recorded presentation is not included in the initial repository.

## Reproducing the analysis

1. Install **Python 3** and Jupyter Notebook or JupyterLab.
2. Download the BankSim dataset from its authorized source. Place `bs140513_032310.csv` alongside the notebook, or adjust the notebook's `pd.read_csv(...)` path to the CSV's location.
3. Open `fraud_detection_analysis.ipynb` and inspect its import and setup cells for required packages. The project uses Python scientific-computing and machine-learning libraries, including **pandas** and **scikit-learn**.
4. Run the notebook cells in order, allowing time for cross-validation and hyperparameter searches.
5. Compare generated tables and charts with the accompanying white paper.

**Reproducibility note:** The notebook was developed in an academic environment (the paper reports Python 3.12.4 and scikit-learn 1.8.0). Environment-specific package versions and execution time may affect the ability to reproduce results exactly. A pinned requirements file and clean-environment test have not yet been provided; the repository should not be described as fully reproducible until those checks are completed.

## Responsible use and limitations

- **Synthetic data:** BankSim is not a substitute for actual institutional transaction data. Reported scores should not be presented as expected real-world fraud-detection performance.
- **Simulation artifacts:** The dataset contains structured patterns, including a fixed number of fraud cases per simulation step, that may not generalize.
- **Fairness:** Coded demographic subgroup checks identify areas to investigate, not proof of fairness or discrimination in a real population.
- **Privacy and governance:** Real-world applications would require access controls, data minimization, documented monitoring, and appropriate legal and regulatory review.
- **Human oversight:** Model alerts should prompt review rather than automatically determine that a customer committed fraud.

## Author and portfolio

**Criselvi Lindsey** — Data & Analytics Leadership, Internal Audit Analytics, Business Intelligence, and Applied Data Science

- [Professional portfolio](https://crisevilindsey.github.io/)
- [GitHub profile](https://github.com/crisevilindsey)
- [LinkedIn](https://www.linkedin.com/in/criselvilindsey/)

## Academic context and references

Developed for **DSC 680, Bellevue University**, as *Project 1: Predictive Analytics for Financial Transaction Fraud Detection*. The detailed white paper contains the full literature references, supporting tables, charts, and ethical assessment. The underlying BankSim research is attributed to López-Rojas and Axelsson (2014).

