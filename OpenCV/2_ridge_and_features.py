import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('ames_clean.csv')
print(f"Размер данных: {df.shape}")
print(f"Целевая переменная: SalePrice")
print()


# X (ПРИЗНАКИ) И y (ЦЕЛЬ)
y = df['SalePrice']
X = df.drop('SalePrice', axis=1)
print(f"Количество признаков: {X.shape[1]}")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Обучающая выборка: {X_train.shape[0]} домов")
print(f"Тестовая выборка: {X_test.shape[0]} домов")
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Признаки нормализованы (среднее=0, дисперсия=1)")
print()


print(" ")
print("ОБУЧЕНИЕ RIDGE РЕГРЕССИИ")
print(" ")

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

# Предсказания
y_pred_train = ridge.predict(X_train_scaled)
y_pred_test = ridge.predict(X_test_scaled)

# Метрики качества
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Обучающая выборка:")
print(f"  MSE: {train_mse:,.0f}")
print(f"  R²: {train_r2:.4f}")
print()
print(f"Тестовая выборка:")
print(f"  MSE: {test_mse:,.0f}")
print(f"  R²: {test_r2:.4f}")
print()


print(" ")
print("ТОП-10 ВАЖНЫХ ПРИЗНАКОВ (по модулю коэффициента)")
print(" ")

# Получаем коэффициенты (веса признаков)
coefficients = ridge.coef_
feature_names = X.columns

# Создаём DataFrame с признаками и их важностью
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
})

# Сортируем по убыванию абсолютного коэффициента
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

# Выводим топ-10
print("\n{:<5} {:<40} {:>15} {:>15}".format('№', 'Признак', 'Коэф-т', '|Коэф-т|'))
print("-" * 80)

for i, row in feature_importance.head(10).iterrows():
    print("{:<5} {:<40} {:>15,.2f} {:>15,.2f}".format(
        i+1, 
        row['feature'][:40], 
        row['coefficient'], 
        row['abs_coefficient']
    ))

print(" ")
print("ТОП-5 ПРИЗНАКОВ, ПОВЫШАЮЩИХ ЦЕНУ")
print(" ")

positive = feature_importance[feature_importance['coefficient'] > 0].head(5)
for i, row in positive.iterrows():
    print(f"{i}. {row['feature'][:45]}: +{row['coefficient']:,.2f}")

print("\n" + " ")
print("ТОП-5 ПРИЗНАКОВ, ПОНИЖАЮЩИХ ЦЕНУ")
print(" ")

negative = feature_importance[feature_importance['coefficient'] < 0].head(5)
for i, row in negative.iterrows():
    print(f"{i}. {row['feature'][:45]}: {row['coefficient']:,.2f}")

print()


# ВИЗУАЛИЗАЦИЯ ТОП-10 ПРИЗНАКОВ
plt.figure(figsize=(10, 8))

top_features = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]

plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Коэффициент (влияние на цену)')
plt.title('Топ-10 самых важных признаков для Ridge регрессии\n(зелёный = повышает цену, красный = понижает)')
plt.gca().invert_yaxis()  # Чтобы самый важный был сверху
plt.tight_layout()

# Сохраняем график
plt.savefig('top_10_features.png', dpi=100)
print("✅ График сохранён как 'top_10_features.png'")
plt.show()

# Сохраняем важность признаков в CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("\n✅ Таблица важности признаков сохранена в 'feature_importance.csv'")