# CVD Uncertainty Screening

Code repository for the paper:

**"Use of Predictive Uncertainty to Improve Risk-Based Decision-Making 
in Cardiovascular Disease Screening"**
Lisa Tabatchnikov & Florian van Daalen
Expert Systems with Applications, Elsevier (under review)

---

## What this project does

This repository contains the full analysis pipeline from the paper.
We apply a Support Vector Classifier (SVC) combined with the 
Multinomial Classification Certainty (MCC) metric to evaluate whether
uncertainty-informed screening reduces missed high-risk CVD cases
while remaining cost-effective.

---

## Project structure

cvd-screening/
├── README.md
├── requirements.txt
├── cvd_dataset.csv
├── Final_Version_Thesis.ipynb
├── src/
│   ├── config.py            all settings and cost parameters
│   ├── mcc.py               MCC metric implementation
│   ├── zones.py             zone classification logic
│   ├── cost_analysis.py     cost-effectiveness analysis
│   └── preprocessing.py     data loading and preprocessing
└── results/                 generated figures and tables

---

## How to run

1. Install dependencies:
   pip install -r requirements.txt

2. Place the dataset file cvd_dataset.csv in the cvd-screening folder

3. Open and run the notebook:
   Final_Version_Thesis.ipynb

---

## Dependencies

pandas, numpy, scikit-learn, scipy, matplotlib
See requirements.txt for version details.

---

## Dataset

CAIR-CVD-2025 — Bangladeshi CVD patient dataset (n=1,529)
Source: Nirob et al., 2025

---

## Contact

Lisa Tabatchnikov
Maastricht University, Faculty of Health, Medicine and Life Sciences