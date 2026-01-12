import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('telecom_customer_churn_encoded_merged.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']

# 加 stratify，確保比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

train_size = X_train.shape[0]
test_size = X_test.shape[0]

print(f"訓練集資料大小: {train_size}")
print(f"測試集資料大小: {test_size}")

# 計算流失與非流失
train_counts = y_train.value_counts().sort_index()  # 0: 未流失, 1: 流失
test_counts = y_test.value_counts().sort_index()

print("訓練集 Churn 分布:")
print(train_counts)
print("測試集 Churn 分布:")
print(test_counts)

sizes = [train_size, test_size]
labels = ['Training Set (80%)', 'Test Set (20%)']
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

# 訓練集、測試集的總數
plt.text(-1.2, 1, f'{train_size} records\n0: {train_counts[0]}\n1: {train_counts[1]}', 
         fontsize=12, color='black', ha='center')
plt.text(1.2, -1, f'{test_size} records\n0: {test_counts[0]}\n1: {test_counts[1]}', 
         fontsize=12, color='black', ha='center')

plt.title(f'Data Split: {train_size} Training Records vs {test_size} Test Records')
plt.axis('equal')
plt.show()

# 合併 X 和 y，避免 SettingWithCopyWarning
train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("訓練集和測試集已儲存為 'train_data.csv' 和 'test_data.csv'.")