import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('telecom_customer_churn_encoded_merged.csv')


# 計算性別與流失狀況的交叉表1
cross_tab = pd.crosstab(df['gender'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: gender vs Churns')
plt.xlabel('gender')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()

# 計算性別與流失狀況的交叉表2
cross_tab = pd.crosstab(df['SeniorCitizen'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: SeniorCitizen vs Churns')
plt.xlabel('SeniorCitizen')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()

# 計算性別與流失狀況的交叉表3
cross_tab = pd.crosstab(df['Partner'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: Partner vs Churns')
plt.xlabel('Partner')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()

# 計算性別與流失狀況的交叉表4
cross_tab = pd.crosstab(df['Dependents'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: Dependents vs Churns')
plt.xlabel('Dependents')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()

# 計算性別與流失狀況的交叉表5
cross_tab = pd.crosstab(df['New_InternetService'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: New_InternetService vs Churns')
plt.xlabel('New_InternetService')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()

# 計算性別與流失狀況的交叉表6
cross_tab = pd.crosstab(df['PaymentMethod'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: PaymentMethod vs Churns')
plt.xlabel('PaymentMethod')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()

# 計算性別與流失狀況的交叉表7
cross_tab = pd.crosstab(df['Contract'], df['Churn'])
# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: Contract vs Churns')
plt.xlabel('Contract')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()


# 繪製折線圖1
# 設定 tenure 區間
bins = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72] # 0-12, 13-36, 37-60, 61-72
labels = ['4', '8', '12', '16', '20', '24', '28', '32', '36', '40', '44', '48', '52', '56', '60', '64', '68', '72']

# 創建一個新的類別欄位 'tenure'
df['tenure'] = pd.cut(
    df['tenure'], 
    bins=bins, 
    labels=labels, 
    right=True,
    include_lowest=True
)

# 分組計數並轉換為寬格式
# 1. 同時對 'tenure' 和 'Churn' 進行分組，然後計數
grouped_counts = df.groupby(['tenure', 'Churn']).size().reset_index(name='Count')
# 2. 使用 pivot_table 將 'Churn' 的 0 和 1 變成兩欄 (寬格式)
pivot_df = grouped_counts.pivot_table(
    index='tenure', 
    columns='Churn', 
    values='Count', 
    fill_value=0 # 如果某個組合沒有數據，填 0
)

# 繪製雙線折線圖
plt.figure(figsize=(12, 7)) 
# 繪製第一條線 (Churn = 0, 即「未流失」用戶)
# pivot_df[0] 欄位
plt.plot(pivot_df.index, pivot_df[0], 
         marker='o', 
         linestyle='-', 
         color='green', 
         label='Churn=0',
         linewidth=2)
# 繪製第二條線 (Churn = 1, 即「已流失」用戶)
# pivot_df[1] 欄位
plt.plot(pivot_df.index, pivot_df[1], 
         marker='s', 
         linestyle='--', 
         color='red', 
         label='Churn=1',
         linewidth=2)

# 添加圖表元素
plt.title('Count of people by tenure', fontsize=16)
plt.xlabel('tenure', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.legend(title='', fontsize=10, loc=1)
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle=':') 
plt.tight_layout()
plt.show()


# 繪製折線圖2
# 設定 tenure 區間
bins = [0, 20, 40, 60, 80, 100, 120, 140] # 0-12, 13-36, 37-60, 61-72
labels = ['20', '40', '60', '80', '100', '120', '140']

# 創建一個新的類別欄位 'MonthlyCharges_Group'
df['MonthlyCharges_Group'] = pd.cut(
    df['MonthlyCharges'], 
    bins=bins, 
    labels=labels, 
    right=True,
    include_lowest=True
)

# 分組計數並轉換為寬格式
# 1. 同時對 'MonthlyCharges_Group' 和 'Churn' 進行分組，然後計數
grouped_counts = df.groupby(['MonthlyCharges_Group', 'Churn']).size().reset_index(name='Count')
# 2. 使用 pivot_table 將 'Churn' 的 0 和 1 變成兩欄 (寬格式)
pivot_df = grouped_counts.pivot_table(
    index='MonthlyCharges_Group', 
    columns='Churn', 
    values='Count', 
    fill_value=0 # 如果某個組合沒有數據，填 0
)

# 繪製雙線折線圖
plt.figure(figsize=(12, 7)) 
# 繪製第一條線 (Churn = 0, 即「未流失」用戶)
# pivot_df[0] 欄位
plt.plot(pivot_df.index, pivot_df[0], 
         marker='o', 
         linestyle='-', 
         color='green', 
         label='Churn=0',
         linewidth=2)
# 繪製第二條線 (Churn = 1, 即「已流失」用戶)
# pivot_df[1] 欄位
plt.plot(pivot_df.index, pivot_df[1], 
         marker='s', 
         linestyle='--', 
         color='red', 
         label='Churn=1',
         linewidth=2)

# 添加圖表元素
plt.title('Count of people by MonthlyCharges', fontsize=16)
plt.xlabel('MonthlyCharges', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.legend(title='', fontsize=10, loc=1)
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle=':') 
plt.tight_layout()
plt.show()