import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.data_processing import prepare_data

# -----------------------------------------------------------------------
# NGƯỠNG MỤC TIÊU — dừng lại khi đạt cả 3
# -----------------------------------------------------------------------
TARGET_F1  = 0.92
TARGET_ACC = 0.90
TARGET_AUC = 0.90

# Số tổ hợp thử cho RandomizedSearchCV (RF và GB)
N_ITER_RF = 60
N_ITER_GB = 80

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def _evaluate(model, X, y):
    """Tính đầy đủ metrics trên tập test."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'accuracy' : float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall'   : float(recall_score(y, y_pred, zero_division=0)),
        'f1'       : float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc'  : float(roc_auc_score(y, y_prob)),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist()
    }


def _print_metrics(name, m, best_params=None):
    print(f"\n  [{name}]")
    print(f"    Accuracy:  {m['accuracy']*100:.2f}%")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}")
    print(f"    F1-Score:  {m['f1']:.4f}")
    print(f"    ROC-AUC:   {m['roc_auc']:.4f}")
    if best_params:
        print(f"    Best params: {best_params}")


def _target_reached(m):
    return (
        m['accuracy'] >= TARGET_ACC and
        m['f1']       >= TARGET_F1  and
        m['roc_auc']  >= TARGET_AUC
    )


# -----------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# -----------------------------------------------------------------------
def train_and_evaluate_models():
    """
    Huấn luyện & tối ưu 4 mô hình ML:
      1. Logistic Regression  — GridSearchCV  (grid nhỏ, nhanh)
      2. Decision Tree        — RandomizedSearchCV
      3. Random Forest        — RandomizedSearchCV  (N_ITER=60)
      4. Gradient Boosting    — RandomizedSearchCV  (N_ITER=80) ← MÔ HÌNH CHÍNH

    Dừng khi mô hình tốt nhất đạt: F1 ≥ 0.92 | Acc ≥ 90% | AUC ≥ 0.90
    """
    print("=" * 60)
    print("   PIPELINE HUẤN LUYỆN & TỐI ƯU MÔ HÌNH")
    print("=" * 60)

    print("\n[STEP 1] Chuẩn bị dữ liệu + Feature Engineering...")
    data = prepare_data()

    X_train, y_train = data['X_train'], data['y_train']
    X_test,  y_test  = data['X_test'],  data['y_test']
    feature_names    = data['feature_names']

    os.makedirs('models', exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # =================================================================
    # MODEL 1: LOGISTIC REGRESSION — GridSearchCV
    # =================================================================
    print("\n[STEP 2] Logistic Regression (GridSearchCV)...")
    lr_grid = {
        'C'     : [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    }
    gs_lr = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'),
        lr_grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True
    )
    gs_lr.fit(X_train, y_train)
    best_lr = gs_lr.best_estimator_
    m_lr = _evaluate(best_lr, X_test, y_test)
    m_lr['best_params'] = gs_lr.best_params_
    results['Logistic Regression'] = m_lr
    joblib.dump(best_lr, 'models/logistic_regression_model.joblib')
    print(f"     F1={m_lr['f1']:.4f} | Acc={m_lr['accuracy']:.4f} | AUC={m_lr['roc_auc']:.4f}")

    # =================================================================
    # MODEL 2: DECISION TREE — RandomizedSearchCV
    # =================================================================
    print("\n[STEP 3] Decision Tree (RandomizedSearchCV)...")
    dt_param = {
        'max_depth'        : [4, 6, 8, 10, 12, 16, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf' : [1, 2, 5, 10],
        'criterion'        : ['gini', 'entropy']
    }
    rs_dt = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        dt_param, n_iter=40, cv=cv, scoring='roc_auc',
        n_jobs=-1, random_state=42, refit=True
    )
    rs_dt.fit(X_train, y_train)
    best_dt = rs_dt.best_estimator_
    m_dt = _evaluate(best_dt, X_test, y_test)
    m_dt['best_params'] = rs_dt.best_params_
    results['Decision Tree'] = m_dt
    joblib.dump(best_dt, 'models/decision_tree_model.joblib')
    print(f"     F1={m_dt['f1']:.4f} | Acc={m_dt['accuracy']:.4f} | AUC={m_dt['roc_auc']:.4f}")

    # =================================================================
    # MODEL 3: RANDOM FOREST — RandomizedSearchCV (N_ITER=60)
    # =================================================================
    print(f"\n[STEP 4] Random Forest (RandomizedSearchCV, n_iter={N_ITER_RF})...")
    rf_param = {
        'n_estimators'     : [100, 200, 300, 400, 500],
        'max_depth'        : [8, 10, 12, 16, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf' : [1, 2, 3, 5],
        'max_features'     : ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap'        : [True, False]
    }
    rs_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        rf_param, n_iter=N_ITER_RF, cv=cv, scoring='roc_auc',
        n_jobs=-1, random_state=42, refit=True, verbose=0
    )
    rs_rf.fit(X_train, y_train)
    best_rf = rs_rf.best_estimator_
    m_rf = _evaluate(best_rf, X_test, y_test)
    m_rf['best_params'] = rs_rf.best_params_
    m_rf['feature_importances'] = sorted(
        zip(feature_names, [float(x) for x in best_rf.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )
    results['Random Forest'] = m_rf
    joblib.dump(best_rf, 'models/random_forest_model.joblib')
    print(f"     F1={m_rf['f1']:.4f} | Acc={m_rf['accuracy']:.4f} | AUC={m_rf['roc_auc']:.4f}")

    # =================================================================
    # MODEL 4: GRADIENT BOOSTING — RandomizedSearchCV (N_ITER=80)
    # =================================================================
    print(f"\n[STEP 5] Gradient Boosting (RandomizedSearchCV, n_iter={N_ITER_GB})...")
    gb_param = {
        'n_estimators'     : [100, 200, 300, 500],
        'learning_rate'    : [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        'max_depth'        : [3, 4, 5, 6, 7],
        'subsample'        : [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 3],
        'max_features'     : ['sqrt', 'log2', None]
    }
    rs_gb = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_param, n_iter=N_ITER_GB, cv=cv, scoring='roc_auc',
        n_jobs=-1, random_state=42, refit=True, verbose=0
    )
    rs_gb.fit(X_train, y_train)
    best_gb = rs_gb.best_estimator_
    m_gb = _evaluate(best_gb, X_test, y_test)
    m_gb['best_params'] = rs_gb.best_params_
    m_gb['feature_importances'] = sorted(
        zip(feature_names, [float(x) for x in best_gb.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )
    results['Gradient Boosting'] = m_gb
    joblib.dump(best_gb, 'models/gradient_boosting_model.joblib')
    print(f"     F1={m_gb['f1']:.4f} | Acc={m_gb['accuracy']:.4f} | AUC={m_gb['roc_auc']:.4f}")

    # =================================================================
    # TỔNG KẾT & KIỂM TRA NGƯỠNG
    # =================================================================
    print("\n" + "=" * 60)
    print("   KẾT QUẢ TỔNG HỢP")
    print("=" * 60)

    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_m = results[best_model_name]

    for name, m in results.items():
        star = "  ★ BEST" if name == best_model_name else ""
        _print_metrics(f"{name}{star}", m, m.get('best_params'))

    print("\n" + "-" * 60)
    print(f"  Mô hình tốt nhất: {best_model_name}")
    print(f"  F1 = {best_m['f1']:.4f}  "
          f"{'✔' if best_m['f1'] >= TARGET_F1 else '✘'} (target ≥ {TARGET_F1})")
    print(f"  Acc= {best_m['accuracy']*100:.2f}%  "
          f"{'✔' if best_m['accuracy'] >= TARGET_ACC else '✘'} (target ≥ {TARGET_ACC*100:.0f}%)")
    print(f"  AUC= {best_m['roc_auc']:.4f}  "
          f"{'✔' if best_m['roc_auc'] >= TARGET_AUC else '✘'} (target ≥ {TARGET_AUC})")

    if _target_reached(best_m):
        print("\n  ✅ ĐẠT NGƯỠNG MỤC TIÊU!")
    else:
        print("\n  ⚠  Chưa đạt ngưỡng — cân nhắc mở rộng N_ITER hoặc bổ sung dữ liệu.")

    # Đánh dấu mô hình tốt nhất trong file JSON (dùng cho app.py)
    results['_best_model'] = best_model_name

    with open('models/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n  Tất cả model đã lưu tại: models/")
    print("  Kết quả đánh giá:        models/evaluation_results.json")
    print("=" * 60)

    return results


if __name__ == '__main__':
    train_and_evaluate_models()