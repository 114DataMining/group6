import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('telecom_customer_churn.csv')

print("每個特徵的資料型態、數量與缺失值數量：")
print(df.info())

# 轉換 TotalCharges欄位為 float64，並處理無法轉換的數據
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("\nTotalCharges 欄位轉換後的資料型態：")
print(df['TotalCharges'].dtype)

# 轉換後再次檢查 TotalCharges 欄位的空值數量
totalcharges_null_count = df['TotalCharges'].isnull().sum()
print(f"\nTotalCharges 欄位的空值數量： {totalcharges_null_count}")

# 若 TotalCharges 欄位有空值，將其填補為 0
if totalcharges_null_count > 0:
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print("\nTotalCharges 欄位空值已填補為 0")

# 轉換後，顯示所有有空值的 customerID
if totalcharges_null_count > 0:
    print("\nTotalCharges 欄位中有空值的 customerID：")
    print(df[df['TotalCharges'] == 0]['customerID'].tolist())


# 1. 檢查數值型特徵的異常值：使用箱型圖檢查數值型資料的異常值
numeric_columns = df.select_dtypes(include=[np.number]).columns  # 選取數值型欄位
# 排除二元類別型特徵 (如 SeniorCitizen, gender 等)
exclude_columns = ['SeniorCitizen', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

# 設定子圖的行數和列數
num_columns = len(numeric_columns)
n_rows = (num_columns // 3) + (1 if num_columns % 3 != 0 else 0)  # 計算行數，最多每行 3 個子圖

# 創建子圖
plt.figure(figsize=(15, 5 * n_rows))

# 顯示每個數值型欄位的異常值箱型圖
for i, column in enumerate(numeric_columns):
    # 計算箱型圖的範圍，超過範圍的資料視為異常值
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  # 四分位距
    lower_bound = Q1 - 1.5 * IQR  
    upper_bound = Q3 + 1.5 * IQR 

    # 篩選異常值
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    # 設定子圖位置
    plt.subplot(n_rows, 3, i + 1)
    sns.boxplot(x=df[column])
    plt.title(f'{column} Boxplot (Outlier)')

    # 顯示異常值範圍及數量
    if outliers.shape[0] > 0:
        print(f"\n{column} (數值型) 異常值數量： {outliers.shape[0]}")
        print(f"{column} 異常值的範圍： ({lower_bound}, {upper_bound})")
        print(f"{column} 異常值範圍內的資料：")
        print(outliers[column].describe())
        print(f"{column} 異常值的部分資料：")
        print(outliers[column].head())
    else:
        print(f"{column} 沒有檢測到異常值。\n")

# 顯示所有的盒鬚圖
plt.tight_layout()
plt.show()


# 2. 檢查類別型特徵的異常值
categorical_columns = df.select_dtypes(include=['object']).columns  # 選取類別型欄位

# 顯示每個類別型欄位的異常值數量
for column in categorical_columns:
    print(f"\n{column} (類別型) 缺失值數量： {df[column].isnull().sum()}")
    
    # 顯示每個類別型欄位的資料類型和分佈
    print(f"{column} 類別型欄位的資料分佈：")
    print(df[column].value_counts(dropna=False))

# 保存處理後的資料集
df.to_csv('telecom_customer_churn_cleaned.csv', index=False)
