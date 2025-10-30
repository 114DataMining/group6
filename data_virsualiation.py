import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('telecom_customer_churn_encoded_merged.csv')


# 繪製長條圖
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='gender', hue='Churn')
plt.title('Gender vs Churns Relationship')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 計算性別與流失狀況的交叉表
cross_tab = pd.crosstab(df['gender'], df['Churn'])


# 繪製堆疊長條圖
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), color=['lightblue', 'orange'])
plt.title('Stacked Bar Chart: Gender vs Churns')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)  # 讓x軸標籤平行顯示
plt.legend(title='Churns', labels=['Not Churned', 'Churned'])
plt.show()