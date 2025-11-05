import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('telecom_customer_churn_cleaned.csv')


# 定義編碼函數
def encode_columns(df):
    # 性別: 男生 1、女性 0
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Partner: Yes 1、No 0
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    
    # Dependents: Yes 1、No 0
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})

    # PhoneService: Yes 1、No 0
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    
    # MultipleLines: No phone service 0、No 1、Yes 2
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
    
    # InternetService: No 0、DSL 1、Fiber optic 2
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    
    # OnlineSecurity、OnlineBackup、DeviceProtection、TechSupport、StreamingTV、StreamingMovies: No internet service 2、No 0、Yes 1
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df[col] = df[col].map({'No internet service': 2, 'No': 0, 'Yes': 1})
    
    # Contract: Month-to-month 0、One year 1、其他 2
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1}).fillna(2)
    
    # PaperlessBilling: Yes 1、No 0
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    
    # PaymentMethod: Electronic check 0、Mailed check 1、Bank transfer (automatic) 2、Credit card (automatic) 3
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    # Churn: Yes 1、No 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

# 編碼
df_encoded = encode_columns(df)

# 計算相關性矩陣
correlation_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
correlation_matrix = df[correlation_cols].corr()

# 設置圖形大小和風格
plt.figure(figsize=(10, 8))
sns.set(style='white')

# 繪製熱力圖
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=1, linecolor='black')
plt.title("Correlation Heatmap between Services")
plt.show()


# 定義合併欄位的函數
def combine_multiple_lines(row):
    if row['PhoneService'] == 0:  # 如果沒有電話服務
        return 0
    elif row['PhoneService'] == 1 and row['MultipleLines'] == 1:  # 只有普通線路
        return 1
    elif row['PhoneService'] == 1 and row['MultipleLines'] == 2:  # 有多條線路
        return 2
    return None

def combine_internet_services(row):
    # 如果沒有網絡服務
    if row['InternetService'] == 0:
        return 0
    # 如果是DSL服務
    elif row['InternetService'] == 1:
        # 當所有服務都為0時，返回1
        if all(service == 0 for service in [row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'], row['StreamingTV'], row['StreamingMovies']]):
            return 1
        # 當有任意服務為1時，返回2
        elif any(service == 1 for service in [row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'], row['StreamingTV'], row['StreamingMovies']]):
            return 2
    # 如果是Fiber optic服務
    elif row['InternetService'] == 2:
        # 當所有服務都為0時，返回3
        if all(service == 0 for service in [row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'], row['StreamingTV'], row['StreamingMovies']]):
            return 3
        # 當有任意服務為1時，返回4
        elif any(service == 1 for service in [row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'], row['StreamingTV'], row['StreamingMovies']]):
            return 4
    return None

# 應用函數來創建新欄位
df['New_MultipleLines'] = df.apply(combine_multiple_lines, axis=1)
df['New_InternetService'] = df.apply(combine_internet_services, axis=1)

plt.figure(figsize=(12, 6))

# New_MultipleLines 長條圖
plt.subplot(1, 2, 1)  # 1行2列，第一個圖
ax1 = sns.countplot(data=df, x='New_MultipleLines', palette='Blues')
plt.title('Distribution of New_MultipleLines')
plt.xlabel('New_MultipleLines')
plt.ylabel('Count')

# 每個條形圖的數量
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

# New_InternetService 長條圖
plt.subplot(1, 2, 2)  # 1行2列，第二個圖
ax2 = sns.countplot(data=df, x='New_InternetService', palette='Blues')
plt.title('Distribution of New_InternetService')
plt.xlabel('New_InternetService')
plt.ylabel('Count')

# 每個條形圖的數量
for p in ax2.patches:
    ax2.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

# 顯示圖形
plt.tight_layout()  # 保證兩個圖不重疊
plt.show()



#儲存
df.to_csv('telecom_customer_churn_encoded_merged.csv', index=False)
print("已重新編碼，儲存在telecom_customer_churn_encoded_merged.csv")