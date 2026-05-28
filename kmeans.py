import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(42)
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.2, random_state=42)
X[:, 0] = X[:, 0] * 8  

k = 3
kmeans = KMeans(n_clusters=k, init='random', n_init=1, max_iter=1, random_state=42)

history_centroids = []
history_labels = []

np.random.seed(42)
initial_indices = np.random.choice(len(X), k, replace=False)
centroids = X[initial_indices].copy()

for iteration in range(10): 
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    history_labels.append(labels.copy())
    history_centroids.append(centroids.copy())
    
    new_centroids = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] 
                              for j in range(k)])
    
    # сходимость
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids


# анимация

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['red', 'blue', 'green']


def update(frame):
    ax.clear()
    labels = history_labels[frame]
    centroids = history_centroids[frame]
    
    for j in range(k):
        cluster_points = X[labels == j]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[j], alpha=0.6, s=40)
    
    ax.scatter(centroids[:, 0], centroids[:, 1], 
               c='black', marker='X', s=200, edgecolors='white', linewidth=2)
    
    #табличка с кластерами и центройдами
    info_text = f"k = {k}\n\nЦентроиды:\n"
    for j, c in enumerate(centroids):
        info_text += f"Кл.{j+1}: ({c[0]:.2f}, {c[1]:.2f})\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    
    ax.set_title(f'K-means, итерация {frame+1} | k = {k}')
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.grid(True, alpha=0.3)


ani = FuncAnimation(fig, update, frames=len(history_labels), repeat=True, interval=800)

# GIF
ani.save('kmeans_animation.gif', writer=PillowWriter(fps=1.5))
print("Гифка сохранена как 'kmeans_animation.gif'")
