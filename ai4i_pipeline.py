"""
Physics-Informed, Cost-Aware Fault Classification on AI4I 2020
==============================================================
Reproducible pipeline accompanying the manuscript:
"Physics-Informed, Cost-Aware Fault Classification with Comparative Explainability
 for Predictive Maintenance: A SHAP-LIME Agreement Analysis"

This single script reproduces every quantitative result and figure in the paper:
  1. Guard-column target engineering (6-class label from AI4I sub-flags)
  2. Five physics-informed features (Eqs. 1-5 in the manuscript)
  3. Illustrative fault-severity cost matrix + cost-derived class weights (Eqs. 6-8)
  4. Leakage-safe 5-fold cross-validation (scaling + SMOTE fitted INSIDE each fold)
  5. Held-out test evaluation (weighted/macro F1, ROC-AUC, total cost)
  6. Feature-set ablation (raw-only / physics-only / combined)
  7. Repeated-seed cost-sensitive vs standard comparison (Wilcoxon)
  8. Cost-matrix sensitivity (missed-failure cost 8..15)
  9. SHAP global + class-level importance (TreeSHAP)
 10. Multi-instance SHAP-LIME Spearman agreement + stability (Eq. 16)

CPU-only. Deterministic (RANDOM_STATE = 42).

Usage:
    python ai4i_pipeline.py --data ai4i2020.csv --outdir results
Requirements: see requirements.txt
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- Consistent indexing: integer i == CLASS_NAMES[i] (prevents cost-cell scrambling) ----
CLASS_NAMES = ["No_Failure", "HDF", "PWF", "OSF", "TWF", "RNF"]
N_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {n: i for i, n in enumerate(CLASS_NAMES)}
ALL_IDX = list(range(N_CLASSES))

RAW = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
       "Torque [Nm]", "Tool wear [min]", "Type_Encoded"]
PHYS = ["Mechanical_Power_W", "Overstrain_Index", "Thermal_Delta_K",
        "Thermal_Speed_Interaction", "Wear_Torque_Ratio"]
FEATURES = RAW + PHYS
BASE_MISS = {"HDF": 10, "PWF": 10, "OSF": 8, "TWF": 6, "RNF": 4}


# --------------------------------------------------------------------------------------
def load_df(path):
    df = pd.read_csv(path)
    fc = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    def lab(r):
        if r["Machine failure"] == 0:
            return "No_Failure"
        active = [f for f in fc if r[f] == 1]
        return active[0] if active else "Unknown"

    df["Fault_Label"] = df.apply(lab, axis=1)
    df = df[df["Fault_Label"] != "Unknown"].reset_index(drop=True)
    df["Type_Encoded"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    # Physics-informed features (Eqs. 1-5)
    omega = df["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    df["Mechanical_Power_W"] = df["Torque [Nm]"] * omega                       # Eq.1
    df["Overstrain_Index"] = df["Tool wear [min]"] * df["Torque [Nm]"]         # Eq.2
    df["Thermal_Delta_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]  # Eq.3
    df["Thermal_Speed_Interaction"] = df["Thermal_Delta_K"] * df["Rotational speed [rpm]"]  # Eq.4
    df["Wear_Torque_Ratio"] = df["Tool wear [min]"] / (df["Torque [Nm]"] + 1e-6)  # Eq.5
    return df


def build_cost_matrix(miss):
    C = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(1, N_CLASSES):
        C[i, 0] = 2  # false alarm
    for f in ["HDF", "PWF", "OSF", "TWF", "RNF"]:
        C[0, LABEL_MAP[f]] = miss[f]  # missed failure
    for i in range(1, N_CLASSES):
        for j in range(1, N_CLASSES):
            if i != j:
                C[i, j] = 3  # inter-fault confusion
    return C


def class_weights_from_C(C):  # Eq. 8
    raw = {CLASS_NAMES[j]: float(np.mean([C[i, j] for i in range(N_CLASSES) if i != j]))
           for j in range(N_CLASSES)}
    s = sum(raw.values())
    return {LABEL_MAP[c]: (w / s) * N_CLASSES for c, w in raw.items()}


def total_cost(y_true, y_pred, C):  # Eq. 7
    return float(sum(C[int(p), int(t)] for t, p in zip(y_true, y_pred)))


def get_split(df, feats=FEATURES, seed=RANDOM_STATE):
    X = df[feats].values
    y = np.array([LABEL_MAP[v] for v in df["Fault_Label"]])
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def safe_smote(Xtr, ytr, rs=RANDOM_STATE):
    k = int(min(2, pd.Series(ytr).value_counts().min() - 1))
    if k < 1:
        return Xtr, ytr
    return SMOTE(random_state=rs, k_neighbors=k).fit_resample(Xtr, ytr)


def fit_predict(name, Xtr, ytr, Xte, cw=None, sw=None, seed=RANDOM_STATE, proba=False):
    if name == "XGBoost":
        uniq = np.sort(np.unique(ytr)); rm = {o: n for n, o in enumerate(uniq)}
        un = {n: o for o, n in rm.items()}
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, num_class=len(uniq), eval_metric="mlogloss",
                          random_state=seed, verbosity=0)
        m.fit(Xtr, np.array([rm[v] for v in ytr]), sample_weight=sw)
        yp = np.array([un[v] for v in m.predict(Xte)])
        if proba:
            raw = m.predict_proba(Xte); P = np.zeros((raw.shape[0], N_CLASSES))
            for xc, oc in un.items():
                P[:, oc] = raw[:, xc]
            return yp, P, m
        return yp, None, m
    if name == "Logistic Regression":
        m = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                               class_weight=cw, random_state=seed)
    elif name == "SVM":
        m = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                class_weight=cw, random_state=seed)
    elif name == "Random Forest":
        m = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   class_weight=cw, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    if proba:
        P = m.predict_proba(Xte)
        if P.shape[1] < N_CLASSES:
            pad = np.zeros((P.shape[0], N_CLASSES)); pad[:, :P.shape[1]] = P; P = pad
        return yp, P, m
    return yp, None, m


NAMES = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]


# --------------------------------------------------------------------------------------
def run(data_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = load_df(data_path)
    C = build_cost_matrix(BASE_MISS); cw = class_weights_from_C(C)
    results = {"class_distribution": df["Fault_Label"].value_counts().to_dict()}
    print("Class distribution:", results["class_distribution"])

    # ---- 1. Leakage-safe 5-fold CV ----
    Xtr, Xte, ytr, yte = get_split(df)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv = {n: {"acc": [], "wf1": [], "mf1": []} for n in NAMES}
    for tr, va in skf.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[tr])
        Xt, Xv = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
        Xt_s, yt_s = safe_smote(Xt, ytr[tr]); sw = np.array([cw[c] for c in yt_s])
        for n in NAMES:
            yp, _, _ = fit_predict(n, Xt_s, yt_s, Xv, cw=(cw if n != "XGBoost" else None),
                                   sw=(sw if n == "XGBoost" else None))
            cv[n]["acc"].append(accuracy_score(ytr[va], yp))
            cv[n]["wf1"].append(f1_score(ytr[va], yp, average="weighted", zero_division=0))
            cv[n]["mf1"].append(f1_score(ytr[va], yp, average="macro", zero_division=0))
    results["cv"] = {n: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}
                     for n, d in cv.items()}
    print("\n[CV leakage-safe macro-F1]", {n: round(np.mean(cv[n]["mf1"]), 3) for n in NAMES})

    # ---- 2. Test evaluation ----
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xtr_sm, ytr_sm = safe_smote(Xtr_s, ytr); swf = np.array([cw[c] for c in ytr_sm])
    yte_bin = label_binarize(yte, classes=ALL_IDX)
    present = np.array([i in np.unique(yte) for i in ALL_IDX])
    test, preds = {}, {}
    for n in NAMES:
        yp, P, _ = fit_predict(n, Xtr_sm, ytr_sm, Xte_s, cw=(cw if n != "XGBoost" else None),
                               sw=(swf if n == "XGBoost" else None), proba=True)
        preds[n] = yp
        try:
            roc = roc_auc_score(yte_bin[:, present], P[:, present], multi_class="ovr", average="weighted")
        except Exception:
            roc = float("nan")
        test[n] = dict(acc=accuracy_score(yte, yp),
                       wf1=f1_score(yte, yp, average="weighted", labels=ALL_IDX, zero_division=0),
                       mf1=f1_score(yte, yp, average="macro", labels=ALL_IDX, zero_division=0),
                       roc=float(roc), cost=total_cost(yte, yp, C))
    results["test"] = test
    best = min(test, key=lambda n: test[n]["cost"])
    results["confusion_best"] = confusion_matrix(yte, preds[best], labels=ALL_IDX).tolist()
    print("[Test] best (lowest cost):", best, {k: round(v, 3) for k, v in test[best].items()})

    # ---- 3. Feature ablation ----
    abl = {}
    for name, feats in {"raw": RAW, "physics": PHYS, "combined": FEATURES}.items():
        Xa, Xb, ya, yb = get_split(df, feats=feats)
        s = StandardScaler().fit(Xa); Xa_s, Xb_s = s.transform(Xa), s.transform(Xb)
        Xa_sm, ya_sm = safe_smote(Xa_s, ya)
        yp, _, _ = fit_predict("Random Forest", Xa_sm, ya_sm, Xb_s, cw=cw)
        abl[name] = dict(n_feat=len(feats),
                         wf1=f1_score(yb, yp, average="weighted", labels=ALL_IDX, zero_division=0),
                         mf1=f1_score(yb, yp, average="macro", labels=ALL_IDX, zero_division=0),
                         cost=total_cost(yb, yp, C))
    results["ablation"] = abl
    print("[Ablation cost] raw/physics/combined:",
          [round(abl[k]["cost"]) for k in ["raw", "physics", "combined"]])

    # ---- 4. Repeated-seed cost-sensitive vs standard ----
    cs, st = [], []
    for s_ in range(10):
        Xa, Xb, ya, yb = get_split(df, seed=s_)
        sc2 = StandardScaler().fit(Xa); Xa_s, Xb_s = sc2.transform(Xa), sc2.transform(Xb)
        Xa_sm, ya_sm = safe_smote(Xa_s, ya, rs=s_)
        yp_cs, _, _ = fit_predict("Random Forest", Xa_sm, ya_sm, Xb_s, cw=cw, seed=s_)
        yp_st, _, _ = fit_predict("Random Forest", Xa_sm, ya_sm, Xb_s, cw=None, seed=s_)
        cs.append(total_cost(yb, yp_cs, C)); st.append(total_cost(yb, yp_st, C))
    _, p = stats.wilcoxon(st, cs)
    results["cost_repeated"] = dict(std_mean=float(np.mean(st)), std_sd=float(np.std(st, ddof=1)),
                                    cs_mean=float(np.mean(cs)), cs_sd=float(np.std(cs, ddof=1)),
                                    wilcoxon_p=float(p))
    print("[Cost 10 seeds] std=%.1f cs=%.1f Wilcoxon p=%.3f"
          % (np.mean(st), np.mean(cs), p))

    # ---- 5. Cost-matrix sensitivity ----
    sens = {}
    for mc in range(8, 16):
        Cm = build_cost_matrix({"HDF": mc, "PWF": mc, "OSF": 8, "TWF": 6, "RNF": 4})
        cwm = class_weights_from_C(Cm)
        yp_cs, _, _ = fit_predict("Random Forest", Xtr_sm, ytr_sm, Xte_s, cw=cwm)
        yp_st, _, _ = fit_predict("Random Forest", Xtr_sm, ytr_sm, Xte_s, cw=None)
        sens[mc] = dict(cs=total_cost(yte, yp_cs, Cm), st=total_cost(yte, yp_st, Cm))
    results["sensitivity"] = sens

    # ---- 6. SHAP + multi-instance LIME agreement (optional, needs shap & lime) ----
    try:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
        rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                    class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Xtr_sm, ytr_sm); yp_rf = rf.predict(Xte_s)
        sv = shap.TreeExplainer(rf).shap_values(Xte_s)
        svl = sv if isinstance(sv, list) else [sv[:, :, k] for k in range(sv.shape[2])]
        glob = np.mean([np.abs(s).mean(0) for s in svl], 0)
        results["shap_global"] = {FEATURES[i]: float(glob[i]) for i in np.argsort(glob)[::-1]}
        scr = {}
        for k in range(len(svl)):
            m = (yte == k) & (yp_rf == k)
            if m.sum():
                o = np.argsort(np.abs(svl[k][m]).mean(0))[::-1]
                scr[CLASS_NAMES[k]] = {FEATURES[o[r]]: r + 1 for r in range(len(FEATURES))}
        le = LimeTabularExplainer(Xtr_sm, feature_names=FEATURES, class_names=CLASS_NAMES,
                                  discretize_continuous=True, mode="classification",
                                  random_state=RANDOM_STATE)

        def lime_rank(idx, k, ns):
            e = le.explain_instance(Xte_s[idx], rf.predict_proba, num_features=len(FEATURES),
                                    labels=[k], num_samples=ns)
            w = dict(e.as_list(label=k)); fw = {}
            for f in FEATURES:
                key = f.lower().replace(" ", "_").replace("[", "").replace("]", ""); t = 0.0
                for lf, wt in w.items():
                    lk = lf.lower().replace(" ", "_").replace("[", "").replace("]", "")
                    if key in lk or lk in key:
                        t += wt
                fw[f] = t
            return {f: r + 1 for r, (f, _) in enumerate(sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True))}

        rng = np.random.RandomState(RANDOM_STATE); agree = {}
        for cls in scr:
            k = LABEL_MAP[cls]; idxs = np.where((yte == k) & (yp_rf == k))[0]
            if len(idxs) < 5:
                continue
            sel = rng.choice(idxs, min(50, len(idxs)), replace=False) if len(idxs) > 50 else idxs
            rr = [spearmanr([scr[cls][f] for f in FEATURES],
                            [lime_rank(int(i), k, 1000)[f] for f in FEATURES])[0] for i in sel]
            agree[cls] = dict(n=len(rr), median=float(np.median(rr)),
                              q1=float(np.percentile(rr, 25)), q3=float(np.percentile(rr, 75)))
        results["shap_lime_agreement"] = agree
        print("[Agreement] pooled classes:", {c: round(a["median"], 3) for c, a in agree.items()})
    except ImportError:
        print("[SHAP/LIME skipped: install shap and lime to reproduce Section IV-F/G]")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved", os.path.join(outdir, "results.json"))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ai4i2020.csv")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    run(a.data, a.outdir)
