from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

df = pd.read_csv(r'./path')

vectorizer = TfidfVectorizer(max_features=5000)  # 可根据需要调整
X_tfidf = vectorizer.fit_transform(df['finding'])

scaler = StandardScaler(with_mean=False)  # 注意：TfidfVectorizer 产生的数据已经是稀疏矩阵
X_scaled = scaler.fit_transform(X_tfidf)

# 定义每个类别的样本数
samples_per_category = 20

categories = df['label'].unique()
typical_samples = []

for category in categories:
    print(f"Processing category: {category}")
    
    category_data = X_scaled[df['label'] == category]
    category_indices = df.index[df['label'] == category]
    
    # 增加k值以捕捉更多模式
    k = min(20, len(category_indices))  # 避免簇数超过样本数
    print(f"Clustering into {k} clusters for category {category}")
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(category_data)
    
    cluster_sizes = np.bincount(kmeans.labels_)
    
    sample_allocation = (cluster_sizes / len(category_indices) * samples_per_category).astype(int)
    
    if sum(sample_allocation) < samples_per_category:
        remaining = samples_per_category - sum(sample_allocation)
        for i in range(len(sample_allocation)):
            if remaining <= 0:
                break
            sample_allocation[i] += 1
            remaining -= 1
    
    # 挑选典型样本
    for i in range(k):
        if sample_allocation[i] > 0:
            cluster_samples = category_data[kmeans.labels_ == i].toarray()
            if len(cluster_samples) > 0:
                distances = np.linalg.norm(cluster_samples - kmeans.cluster_centers_[i], axis=1)
                closest_sample_indices = np.argsort(distances)[:sample_allocation[i]]
                typical_samples.extend(category_indices[kmeans.labels_ == i][closest_sample_indices])


typical_samples_df = df.loc[typical_samples]


print("Typical samples:")
print(typical_samples_df)
csv_output_path = r'__'
typical_samples_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")


txt_output_path = r'clustertypical_samples_80.txt'
with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
    for _, row in typical_samples_df.iterrows():
        line = f"label: {row['label']}, finding: {row['finding']}\n"
        txt_file.write(line)

print(f"Typical samples have been saved to CSV and TXT files at {csv_output_path} and {txt_output_path}.")