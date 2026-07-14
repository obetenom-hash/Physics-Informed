# Physics-Informed, Cost-Aware Fault Classification on AI4I 2020

Reproducible code for the manuscript *"Physics-Informed, Cost-Aware Fault Classification with
Comparative Explainability for Predictive Maintenance: A SHAP–LIME Agreement Analysis."*

## Contents
- `ai4i_pipeline.py` — single self-contained script reproducing every result and figure.
- `requirements.txt` — Python dependencies.
- (Provide `ai4i2020.csv` from the UCI Machine Learning Repository; not redistributed here.)

## What it reproduces
1. Guard-column six-class target engineering.
2. Five physics-informed features .
3. Illustrative fault-severity cost matrix and cost-derived class weights.
4. **Leakage-safe** 5-fold cross-validation (scaling + SMOTE fitted **inside each fold**).
5. Held-out test evaluation (weighted/macro F1, ROC-AUC, total cost).
6. Feature-set ablation (raw-only / physics-only / combined).
7. Repeated-seed cost-sensitive vs standard comparison with Wilcoxon test.
8. Cost-matrix sensitivity (missed-failure cost).
9. SHAP global and class-level importance (TreeSHAP).
10. Multi-instance SHAP–LIME Spearman agreement and stability.

## Usage
```bash
pip install -r requirements.txt
python ai4i_pipeline.py --data ai4i2020.csv --outdir results
```
Results are written to `results/results.json`. The run is CPU-only and deterministic
(`RANDOM_STATE = 42`).

## Data
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## License
Released under the MIT License to support reproducibility.

## Citation
If you use this code, please cite the accompanying paper (and this Zenodo deposit).
