import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('telecom_customer_churn_encoded_merged.csv')

# 設Churn是目標變數，並且其他的都是features
X = df.drop('Churn', axis=1)  # 其他所有欄位是特徵
y = df['Churn']  # 'Churn' 是目標變數

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 計算訓練集和測試集的筆數
train_size = X_train.shape[0]
test_size = X_test.shape[0]

# 顯示資料切割後的資料大小
print(f"訓練集資料大小: {X_train.shape[0]}")
print(f"測試集資料大小: {X_test.shape[0]}")


# 計算總資料筆數
total_size = train_size + test_size

# 生成圓餅圖的資料
sizes = [train_size, test_size]
labels = ['Training Set (80%)', 'Test Set (20%)']
colors = ['#66b3ff', '#ff9999']

# 繪製圓餅圖
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

# 在圓餅圖上標註資料筆數
plt.text(-1.2, 1, f'{train_size} records', fontsize=12, color='black', ha='center')
plt.text(1.2, -1, f'{test_size} records', fontsize=12, color='black', ha='center')

# 圓餅圖標題
plt.title(f'Data Split: {train_size} Training Records vs {test_size} Test Records')

# 保證圓餅圖為圓形
plt.axis('equal')

plt.show()


# 儲存訓練集和測試集為
X_train['Churn'] = y_train  # 將目標變數加入訓練集
X_test['Churn'] = y_test    # 將目標變數加入測試集

X_train.to_csv('train_data.csv', index=False)
X_test.to_csv('test_data.csv', index=False)

print("訓練集和測試集已儲存為 'train_data.csv' 和 'test_data.csv'.")