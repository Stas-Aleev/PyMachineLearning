import random
import math
import matplotlib.pyplot as plt

def generate_random_cities(n_cities, width=100, height=100):
    cities = []
    for i in range(n_cities):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        cities.append((x, y))
    return cities

cities = generate_random_cities(30)

def distance(city1, city2):
    dx = city1[0] - city2[0]
    dy = city1[1] - city2[1]
    return math.sqrt(dx*dx + dy*dy)

def total_distance(route):
    total = 0
    for i in range(len(route)):
        city_a = cities[route[i]]
        city_b = cities[route[(i+1) % len(route)]]
        total += distance(city_a, city_b)
    return total

def plot_route(route, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    x_coords = [cities[city][0] for city in route]
    y_coords = [cities[city][1] for city in route]
    x_coords.append(cities[route[0]][0])
    y_coords.append(cities[route[0]][1])
    
    ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
    ax.scatter(x_coords[:-1], y_coords[:-1], c='red', s=100, zorder=5)
    
    for i, city in enumerate(route):
        ax.annotate(str(i), (cities[city][0]+0.1, cities[city][1]+0.1), fontsize=12)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X координата')
    ax.set_ylabel('Y координата')
    ax.grid(True, alpha=0.3)
    return ax

def simulated_annealing(initial_temp=100, cooling_rate=0.39, iter=5000):
    current_route = list(range(len(cities)))
    random.shuffle(current_route)
    current_length = total_distance(current_route)
    
    best_route = current_route[:]
    best_length = current_length
    
    history_lengths = [current_length]
    history_temp = [initial_temp]
    
    t = initial_temp
    print("Начальный маршрут:", current_route)
    print("Длина начального маршрута:", round(current_length, 2))

    for step in range(iter):
        new_route = current_route[:]
        i, j = random.sample(range(len(cities)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        
        new_length = total_distance(new_route)
        delta = new_length - current_length 
        
        if delta < 0:
            accept = True
        else:
            probability = math.exp(-delta / t)
            accept = random.random() < probability
        
        if accept:
            current_route = new_route
            current_length = new_length
            if current_length < best_length:
                best_route = current_route[:]
                best_length = current_length
        
        t *= cooling_rate
        history_lengths.append(best_length)
        history_temp.append(t)
        
        if step % 200 == 0 and step > 0:
            print(f"  Шаг {step}: текущая длина = {current_length:.2f}, "
                  f"лучшая = {best_length:.2f}, температура = {t:.2f}")
    
    print(f"\nЛучшая длина: {best_length:.2f}")
    return best_route, best_length, history_lengths, history_temp

best_route, best_length, history_lengths, history_temp = simulated_annealing(
    initial_temp=100,   
    cooling_rate=0.997, 
    iter=3000     
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

start_route = list(range(len(cities)))
random.shuffle(start_route)
plot_route(start_route, f"Начальный маршрут (длина = {total_distance(start_route):.2f})", ax1)
plot_route(best_route, f"Лучший маршрут (длина = {best_length:.2f})", ax2)

plt.tight_layout()
plt.show()

fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))

ax3.plot(history_lengths, 'b-', linewidth=1.5)
ax3.set_title('Улучшение маршрута во времени', fontsize=14)
ax3.set_xlabel('Шаг алгоритма')
ax3.set_ylabel('Длина маршрута')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=best_length, color='r', linestyle='--', 
            label=f'Минимум: {best_length:.2f}')
ax3.legend()

ax4.semilogy(history_temp, 'r-', linewidth=1.5)
ax4.set_title('Падение температуры (логарифмическая шкала)', fontsize=14)
ax4.set_xlabel('Шаг алгоритма')
ax4.set_ylabel('Температура')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
print("="*50)
print(f"Порядок посещения городов: {best_route}")
print(f"Общая длина маршрута: {best_length:.2f}")
print("\nПорядок в виде списка городов (координаты):")
for i, city_idx in enumerate(best_route):
    print(f"  Шаг {i+1}: город {city_idx} {cities[city_idx]}")
