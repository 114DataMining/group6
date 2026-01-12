#過採樣（Oversampling）
#SMOTENC 針對「類別特徵」和「連續特徵」分開處理
#類別特徵 → 不做線性內插，直接從鄰居中隨機選取一個類別值，避免生成 0.3、1.7 這種對類別沒有意義的值
#連續特徵 → 做線性內插（保留數值合理性）

import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train_data.csv')

print("原始 Churn 分布：")
print(train_data['Churn'].value_counts())

#每個特徵值分布
feature_columns = ['gender','SeniorCitizen','Partner','Dependents',
                   'New_MultipleLines','New_InternetService','tenure',
                   'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']

print("\n原始資料特徵值分布（前幾筆）：")
for col in feature_columns:
    print(f"\n{col} 值分布：")
    print(train_data[col].value_counts())



binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling']
multi_class_features = ['New_MultipleLines', 'New_InternetService', 'Contract', 'PaymentMethod']
continuous_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
target = 'Churn'

#長條圖
for col in feature_columns:
    plt.figure(figsize=(6,4))
    if col in continuous_features:
        sns.histplot(train_data[col], bins=20, kde=False)
    else:
        sns.countplot(x=col, data=train_data)
    plt.title(f"Original Data: {col} Distribution")
    plt.show()


# 建立特徵矩陣 X 與目標 y，移除 customerID
X_train = train_data.drop(columns=[target, 'customerID'])
y_train = train_data[target]

# 類別欄位索引
categorical_features_idx = [X_train.columns.get_loc(col) for col in (binary_features + multi_class_features)]


# SMOTENC 過採樣
smote_nc = SMOTENC(categorical_features=categorical_features_idx, random_state=42)
X_res, y_res = smote_nc.fit_resample(X_train, y_train)


print("\n過採樣後 Churn 分布：")
print(Counter(y_res))

# 過採樣後每個特徵值分布
train_after = X_res.copy()
train_after['Churn'] = y_res

print("\n過採樣後特徵值分布（前幾筆）：")
for col in feature_columns:
    if col in train_after.columns:
        print(f"\n{col} 值分布：")
        print(train_after[col].value_counts())
#長條圖
for col in feature_columns:
    plt.figure(figsize=(6,4))
    if col in continuous_features:
        sns.histplot(train_after[col], bins=20, kde=False)
    else:
        sns.countplot(x=col, data=train_after)
    plt.title(f"SMOTE Data: {col} Distribution")
    plt.show()


train_after.to_csv('train_after.csv', index=False)
print("\n過採樣後資料已存成 train_after.csv，總筆數:", len(train_after))
