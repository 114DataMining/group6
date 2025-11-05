import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 使用相對路徑載入檔案
df = pd.read_csv('telecom_customer_churn.csv')

# 檢查每個欄位的缺失值數量
missing_values = df.isnull().sum()

print("每個欄位的缺失值數量：")
print(missing_values)

# 篩選出有缺失值的欄位
missing_columns = missing_values[missing_values > 0]

print("有缺失值的欄位：")
print(missing_columns)

# 計算總缺失值數量
total_missing = missing_columns.sum()
print(f"總缺失值數量: {total_missing}")

# 繪製有缺失值的欄位在填補前的長條圖
plt.figure(figsize=(12, 8))

# 遍歷每個有缺失值的欄位
for i, column in enumerate(missing_columns.index):
    plt.subplot(len(missing_columns), 1, i + 1)
    sns.countplot(data=df, x=column, palette='Set2')
    plt.title(f'{column} (Before filling missing values)')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 顯示填補前的資料分佈
for column in missing_columns.index:
    print(f"\n填補前 {column} 的資料分佈：")
    print(df[column].value_counts(dropna=False))

# 填補類別型欄位的缺失值（使用眾數）
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].fillna(df[column].mode()[0])  # 這裡是填補眾數

# 填補數值型欄位的缺失值（使用平均數）
for column in df.select_dtypes(include=[float, int]).columns:
    df[column] = df[column].fillna(df[column].mean())  # 這裡是填補平均數

# 檢查填補後的缺失值數量
missing_values_after = df.isnull().sum()
missing_columns_after = missing_values_after[missing_values_after > 0]

# 顯示填補後的缺失值數量
print("\n填補缺失值後每個欄位的缺失值數量：")
print(missing_columns_after)

# 顯示填補後的資料分佈
print("\n填補後的資料分佈：")
for column in df.select_dtypes(include=['object']).columns:  # 類別型資料
    print(f"{column} 的資料分佈：")
    print(df[column].value_counts(dropna=False))

df.to_csv('telecom_customer_churn.csv', index=False)