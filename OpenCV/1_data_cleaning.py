import pandas as pd
import numpy as np

df = pd.read_csv('AmesHousing.csv')
print(f"Размер: {df.shape}")
print()

# 'NA' на NaN
df = df.replace('NA', np.nan)

# Категориальные → 'None'
na = ['Alley', 'Mas Vnr Type', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
      'BsmtFin Type 1', 'BsmtFin Type 2', 'Electrical', 'Fireplace Qu',
      'Garage Type', 'Garage Finish']

for col in na:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# Числовые → 0
zero = ['Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
        'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Yr Blt', 
        'Garage Cars', 'Garage Area']

for col in zero:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Lot Frontage
print("Lot Frontage...")
print(f"Пропусков до: {df['Lot Frontage'].isna().sum()}")

df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(
    lambda x: x.fillna(x.median())
)

if df['Lot Frontage'].isna().sum() > 0:
    overall_median = df['Lot Frontage'].median()
    df['Lot Frontage'] = df['Lot Frontage'].fillna(overall_median)

print(f"Пропусков после: {df['Lot Frontage'].isna().sum()}")
print()

print("ONE-HOT ENCODING")
print(" ")

categorical_cols = df.select_dtypes(include=['object', 'string']).columns
print(f"Найдено {len(categorical_cols)} категориальных признаков")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print(f"Размер после кодирования: {df_encoded.shape}")
print(f"Признаков стало: {df_encoded.shape[1]} (было {df.shape[1]})")

df_encoded.to_csv('ames_clean.csv', index=False)
print("\n✅ Файл сохранён в 'ames_clean.csv'")

print("\n" + "="*50)
print("Итого")
print("="*50)
print(f"Домов: {len(df_encoded):,}")
print(f"Признаков: {df_encoded.shape[1]}")
print(f"Цена: от ${df_encoded['SalePrice'].min():,.0f} до ${df_encoded['SalePrice'].max():,.0f}")
print(f"Средняя цена: ${df_encoded['SalePrice'].mean():,.0f}")
