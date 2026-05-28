import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ames_clean.csv')
print(f"Загружено домов: {len(df)}")
print(f"Период: {df['Yr Sold'].min()} - {df['Yr Sold'].max()}")

df = df.copy()

df['Age'] = df['Yr Sold'] - df['Year Built']
df['Years_Since_Remod'] = df['Yr Sold'] - df['Year Remod/Add']

print("\n" + "-"*40)
print("ДИНАМИКА ЦЕН ПО ГОДАМ")
print("-"*40)

yearly = df.groupby('Yr Sold')['SalePrice'].agg(['mean', 'median', 'count'])
yearly.columns = ['Средняя', 'Медианная', 'Кол-во']
print(yearly)

if 2008 in yearly.index and 2007 in yearly.index:
    drop_2008 = ((yearly.loc[2008, 'Средняя'] - yearly.loc[2007, 'Средняя']) / yearly.loc[2007, 'Средняя']) * 100
    print(f"\nПадение в 2008 году: {abs(drop_2008):.1f}%")
    
    if 2010 in yearly.index:
        rec = ((yearly.loc[2010, 'Средняя'] - yearly.loc[2008, 'Средняя']) / yearly.loc[2008, 'Средняя']) * 100
        print(f"Восстановление к 2010: {rec:+.1f}%")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(yearly.index, yearly['Средняя'], 'o-', linewidth=2, markersize=8, label='Средняя цена')
plt.plot(yearly.index, yearly['Медианная'], 's--', linewidth=2, markersize=6, label='Медианная цена')
plt.axvline(x=2008, color='red', linestyle='--', alpha=0.7, label='Кризис 2008')
plt.xlabel('Год продажи')
plt.ylabel('Цена ($)')
plt.title('Динамика цен по годам')
plt.legend()
plt.grid(True, alpha=0.3)

print("\n" + "-"*40)
print("СЕЗОННОСТЬ (по месяцам)")
print("-"*40)

monthly = df.groupby('Mo Sold')['SalePrice'].mean()
monthly.index = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
print(monthly)

best_month = monthly.idxmax()
worst_month = monthly.idxmin()
print(f"\nСамый дорогой месяц: {best_month} (${monthly[best_month]:,.0f})")
print(f"Самый дешёвый месяц: {worst_month} (${monthly[worst_month]:,.0f})")
print(f"Разница: {((monthly[best_month] - monthly[worst_month]) / monthly[worst_month]) * 100:.1f}%")

plt.subplot(2, 2, 2)
bars = plt.bar(monthly.index, monthly.values, color='skyblue', edgecolor='navy')
bars[monthly.values.argmax()].set_color('green')
bars[monthly.values.argmin()].set_color('red')
plt.xlabel('Месяц')
plt.ylabel('Средняя цена ($)')
plt.title('Сезонность цен (самый дорогой месяц — зелёный)')
plt.grid(True, alpha=0.3, axis='y')

print("\n" + "-"*40)
print("ТЕПЛОВАЯ КАРТА (год × месяц)")
print("-"*40)

year_month = df.groupby(['Yr Sold', 'Mo Sold'])['SalePrice'].mean().unstack()
print("Средние цены по годам и месяцам (в тысячах $):")
print(round(year_month / 1000, 1))

plt.subplot(2, 2, 3)
im = plt.imshow(year_month.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
plt.colorbar(im, label='Цена ($)')
plt.xlabel('Месяц')
plt.ylabel('Год')
plt.xticks(range(len(year_month.columns)), year_month.columns)
plt.yticks(range(len(year_month.index)), year_month.index)
plt.title('Тепловая карта цен (год × месяц)')
for i in range(len(year_month.index)):
    for j in range(len(year_month.columns)):
        if not pd.isna(year_month.values[i,j]):
            plt.text(j, i, f'{year_month.values[i,j]/1000:.0f}k', 
                    ha='center', va='center', fontsize=8)

print("\n" + "-"*40)
print("ВЛИЯНИЕ ВОЗРАСТА НА ЦЕНУ")
print("-"*40)

age_groups = pd.cut(df['Age'], bins=[0, 5, 10, 20, 40, 100], 
                    labels=['0-5 лет', '6-10 лет', '11-20 лет', '21-40 лет', '40+ лет'])
age_impact = df.groupby(age_groups)['SalePrice'].mean()
print(age_impact)

corr_age = df['Age'].corr(df['SalePrice'])
print(f"\nКорреляция возраст ↔ цена: {corr_age:.3f}")

plt.subplot(2, 2, 4)
plt.scatter(df['Age'], df['SalePrice'], alpha=0.3, s=20)
z = np.polyfit(df['Age'], df['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(df['Age'].sort_values(), p(df['Age'].sort_values()), 'r-', linewidth=2, label='Тренд')
plt.xlabel('Возраст дома на момент продажи (лет)')
plt.ylabel('Цена ($)')
plt.title('Влияние возраста на цену')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_analysis.png', dpi=150)
plt.show()

print("\n" + "-"*40)
print("ВЛИЯНИЕ РЕМОНТА")
print("-"*40)

remod_groups = pd.cut(df['Years_Since_Remod'], bins=[-1, 2, 5, 10, 20, 100],
                      labels=['0-2 года', '3-5 лет', '6-10 лет', '11-20 лет', '20+ лет'])
remod_impact = df.groupby(remod_groups)['SalePrice'].mean()
print(remod_impact)

has_remod = (df['Year Remod/Add'] != df['Year Built']).astype(int)
remod_yes = df[has_remod == 1]['SalePrice'].mean()
remod_no = df[has_remod == 0]['SalePrice'].mean()
premium = ((remod_yes - remod_no) / remod_no) * 100
print(f"\nДома с ремонтом: ${remod_yes:,.0f}")
print(f"Дома без ремонта: ${remod_no:,.0f}")
print(f"Премия за ремонт: {premium:+.1f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sales_by_month = df['Mo Sold'].value_counts().sort_index()
plt.bar(['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек'], 
        sales_by_month.values, color='coral')
plt.xlabel('Месяц')
plt.ylabel('Количество продаж')
plt.title('Сезонность продаж')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
sales_by_year = df['Yr Sold'].value_counts().sort_index()
plt.bar(sales_by_year.index, sales_by_year.values, color='teal')
plt.xlabel('Год')
plt.ylabel('Количество продаж')
plt.title('Динамика продаж по годам')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sales_volume.png', dpi=100)
plt.show()

print("\n" + "-"*40)
print("ИТОГИ ВРЕМЕННОГО АНАЛИЗА")
print("-"*40)

print(f"""
1. КРИЗИС 2008 ГОДА: падение на {abs(drop_2008):.1f}%

2. СЕЗОННОСТЬ: лучший месяц {best_month}, худший {worst_month}, разница {((monthly[best_month] - monthly[worst_month]) / monthly[worst_month]) * 100:.1f}%

3. ВОЗРАСТ: корреляция {corr_age:.3f}

4. РЕМОНТ: премия {premium:+.1f}%

5. ПИК ПРОДАЖ: {best_month}
""")

with open('time_analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Период: {df['Yr Sold'].min()} - {df['Yr Sold'].max()}\n")
    f.write(f"Всего продаж: {len(df)}\n\n")
    f.write("Цены по годам:\n")
    f.write(yearly.to_string())
    f.write(f"\n\nПадение в 2008: {abs(drop_2008):.1f}%\n")
    f.write(f"Премия за ремонт: {premium:+.1f}%\n")
    f.write(f"Корреляция возраст-цена: {corr_age:.3f}\n")

print("\nСохранено: time_analysis_results.txt")
print("Сохранено: time_analysis.png, sales_volume.png")