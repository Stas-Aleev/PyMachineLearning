import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv('ames_clean.csv')
print(f"Загружено домов: {len(df)}")
df['Price_per_sqft'] = df['SalePrice'] / df['Gr Liv Area']

print("\n" + "="*40)
print("ПОИСК АНОМАЛИЙ")
print("="*40)

features = ['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Garage Area']
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(df[features])

anomalies = df[df['anomaly'] == -1]
print(f"Найдено аномалий: {len(anomalies)}")

suspects = anomalies.nsmallest(5, 'Price_per_sqft')
print("\n(дешёвые при большой площади):")
for _, row in suspects.iterrows():
    print(f"  Цена: ${row['SalePrice']:,.0f} | Площадь: {row['Gr Liv Area']} | Качество: {row['Overall Qual']}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(df[df['anomaly']==1]['Gr Liv Area'], df[df['anomaly']==1]['SalePrice'], alpha=0.5, label='Норма')
plt.scatter(anomalies['Gr Liv Area'], anomalies['SalePrice'], color='red', marker='x', label='Аномалии')
plt.xlabel('Площадь'); plt.ylabel('Цена'); plt.title('Аномалии: Цена vs Площадь'); plt.legend()

plt.subplot(1,2,2)
plt.hist(df[df['anomaly']==1]['Price_per_sqft'], bins=50, alpha=0.7, label='Норма')
plt.hist(anomalies['Price_per_sqft'], bins=20, alpha=0.7, color='red', label='Аномалии')
plt.xlabel('Цена за кв.фут'); plt.ylabel('Кол-во'); plt.title('Распределение цены'); plt.legend()
plt.tight_layout()
plt.savefig('anomalies.png')
plt.show()

print("\n" + " ")
print("СРАВНЕНИЕ МОДЕЛИ Ridge")
print(" ")

def evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaler.fit_transform(X_train), y_train)
    y_pred = ridge.predict(scaler.transform(X_test))
    return r2_score(y_test, y_pred)

X = df.drop(['SalePrice', 'Price_per_sqft', 'anomaly'], axis=1)
y = df['SalePrice']

r2_all = evaluate(X, y)
print(f"Модель на ВСЕХ данных: R² = {r2_all:.4f}")

# Удаляем аномалии
df_clean = df[df['anomaly'] == 1].copy()
X_clean = df_clean.drop(['SalePrice', 'Price_per_sqft', 'anomaly'], axis=1)
y_clean = df_clean['SalePrice']

r2_clean = evaluate(X_clean, y_clean)
print(f"Модель БЕЗ аномалий:  R² = {r2_clean:.4f}")

print(f"\nУлучшение: {(r2_clean - r2_all) * 100:+.2f}%")
print("Модель стала лучше..." if r2_clean > r2_all else "Аномалии не мешали")

print("\n" + " ")
print("КЛАСТЕРИЗАЦИЯ (без цены)")
print(" ")

cluster_feat = ['Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Garage Area', 'Year Built']
X_cluster = df_clean[cluster_feat].copy()

scaler = StandardScaler()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(scaler.fit_transform(X_cluster))

# Краткая характеристика кластеров
stats = df_clean.groupby('Cluster').agg({
    'Gr Liv Area': 'mean',
    'Overall Qual': 'mean',
    'Year Built': 'mean',
    'SalePrice': 'mean'
}).round(0)

print("\nХарактеристики кластеров:")
print(stats)

# Называем кластеры
names = {0: "Стандартные", 1: "Премиум", 2: "Средние", 3: "Эконом", 4: "Новые"}
df_clean['Segment'] = df_clean['Cluster'].map(names)
print("\nСегменты:")
for seg in df_clean['Segment'].unique():
    avg_price = df_clean[df_clean['Segment']==seg]['SalePrice'].mean()
    print(f"  {seg}: {len(df_clean[df_clean['Segment']==seg])} домов, средняя цена ${avg_price:,.0f}")

print("\n" + " ")
print("PCA + Ridge")
print(" ")

# Все числовые
num_cols = df_clean.select_dtypes(include=[np.number]).columns
num_cols = [c for c in num_cols if c not in ['SalePrice', 'Price_per_sqft', 'anomaly', 'Cluster']]

X_pca = df_clean[num_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=10, random_state=42)
X_pca_transformed = pca.fit_transform(X_scaled)

print(f"Было признаков: {X_scaled.shape[1]}")
print(f"Стало компонент: {X_pca_transformed.shape[1]}")
print(f"Объяснено дисперсии: {pca.explained_variance_ratio_.sum():.2%}")

r2_pca = evaluate(X_pca_transformed, y_clean)
print(f"R² на PCA-компонентах: {r2_pca:.4f}")

print("\n")
print("ИТОГИ")
print(" ")
print(f"1. Ridge на всех данных:       R² = {r2_all:.4f}")
print(f"2. Ridge без аномалий:         R² = {r2_clean:.4f}")
print(f"3. Ridge + PCA (10 компонент): R² = {r2_pca:.4f}")
print("\n Лучший: " + ("Без аномалий" if r2_clean == max(r2_all, r2_clean, r2_pca) else "PCA" if r2_pca == max(r2_all, r2_clean, r2_pca) else "Все данные"))

df_clean.to_csv('ames_with_segments.csv', index=False)
print("\nСохранено: ames_with_segments.csv")