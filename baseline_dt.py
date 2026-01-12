import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# 1. 讀取資料
# =========================
train = pd.read_csv("train_after.csv")   # 已處理類別不平衡
test = pd.read_csv("test_data.csv")

TARGET = "Churn"

# =========================
# 2. 空值 & 資料型態處理
# =========================
train.replace(' ', pd.NA, inplace=True)
test.replace(' ', pd.NA, inplace=True)

# 將數值欄位轉成 float
num_cols = ['tenure','MonthlyCharges','TotalCharges']
train[num_cols] = train[num_cols].apply(pd.to_numeric, errors='coerce')
test[num_cols] = test[num_cols].apply(pd.to_numeric, errors='coerce')


# =========================
# 3. 準備特徵與目標
# =========================
X_train = train.drop(TARGET, axis=1)
y_train = train[TARGET]

X_test = test.drop(columns=["customerID", TARGET])
y_test = test[TARGET]

# 確保欄位順序一致
X_test = X_test[X_train.columns]

# =========================
# 4. 建立 Baseline 決策樹
# =========================
baseline_dt = DecisionTreeClassifier(random_state=42)
baseline_dt.fit(X_train, y_train)

# =========================
# 5. 預測與評估
# =========================
y_pred = baseline_dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("===== Baseline Decision Tree Evaluation =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
