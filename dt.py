import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


# =========================
# 1. 讀取資料
# =========================
train = pd.read_csv("train_after.csv")
test = pd.read_csv("test_data.csv")
TARGET = "Churn"

# =========================
# 2. 資料切分
# =========================
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# =========================
# 3. 測試集前處理（型態修正）
# =========================
X_test = X_test[X_train.columns]

X_test['New_MultipleLines'] = (
    pd.to_numeric(X_test['New_MultipleLines'], errors='coerce')
    .fillna(0)
    .astype(int)
)

X_test['TotalCharges'] = (
    pd.to_numeric(X_test['TotalCharges'], errors='coerce')
    .fillna(0.0)
    .astype(float)
)

assert X_test.select_dtypes(include='object').empty

from sklearn.model_selection import StratifiedKFold  # 建議改用分層抽樣

# =========================
# 4. 5-Fold Cross-Validation
# =========================
from sklearn.model_selection import StratifiedKFold # 建議使用分層抽樣以維持 0,1 比例

# 這裡可以根據需求選擇 KFold 或 StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mean_fpr = np.linspace(0, 1, 500)
tprs, aucs = [], []

print("\n===== 5-Fold Cross-Validation Metrics =====")

plt.figure(figsize=(8,6))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 統計該折驗證集的 Churn 數量
    counts = y_val.value_counts()
    c0 = counts.get(0, 0)
    c1 = counts.get(1, 0)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # 依照你指定的格式輸出，並加上 Churn 統計
    print(
        f"Fold {fold}: Accuracy={acc:.4f}, Precision={prec:.4f}, "
        f"Recall={rec:.4f}, F1-score={f1:.4f} "
        f"(Churn 0: {c0}, Churn 1: {c1})"
    )

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    aucs.append(auc(fpr, tpr))

    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

# =========================
# 5. GridSearchCV
# =========================
param_grid = {
    'criterion': ['gini'],
    'max_depth': [1,2,3,4,5,6],
    'min_samples_leaf': [1,2,4],
    'min_samples_split': [2,5,10],
    'splitter': ['best','random'],
    'class_weight': [None,'balanced']
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)

print("\n===== GridSearch Best Parameters =====")
print(grid_dt.best_params_)
print("Best CV F1-score:", grid_dt.best_score_)

# =========================
# 6. 最佳模型訓練
# =========================
best_dt = grid_dt.best_estimator_
best_dt.fit(X_train, y_train)

# =========================
# 6.1 Feature Importance（內建）
# =========================
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_dt.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n===== Feature Importance (Decision Tree) =====")
print(feature_importance)

plt.figure(figsize=(10,6))
plt.barh(
    feature_importance['Feature'],
    feature_importance['Importance']
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance - Decision Tree")
plt.grid(True)
plt.show()

# =========================
# 6.2 Permutation Importance
# =========================
perm_result = permutation_importance(
    best_dt,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='f1'
)

perm_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_result.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\n===== Permutation Feature Importance =====")
print(perm_importance)

plt.figure(figsize=(10,6))
plt.barh(
    perm_importance['Feature'],
    perm_importance['Importance']
)
plt.gca().invert_yaxis()
plt.xlabel("Decrease in F1-score")
plt.title("Permutation Feature Importance (Test Set)")
plt.grid(True)
plt.show()


# =========================
# 7. 評估 + 交叉表（新增）
# =========================
def evaluate_model_with_crosstab(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # 交叉表（你指定的格式）
    crosstab = pd.DataFrame(
        [[tn, fn],
         [fp, tp]],
        index=["Predicted Non-Churn", "Predicted Churn"],
        columns=["Actual Non-Churn", "Actual Churn"]
    )

    print("\nConfusion Table:")
    print(crosstab)

    # ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# =========================
# 8. Training / Test
# =========================
evaluate_model_with_crosstab(best_dt, X_train, y_train, "Training Set")
evaluate_model_with_crosstab(best_dt, X_test, y_test, "Test Set")
