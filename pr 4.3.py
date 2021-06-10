import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors # Ввод библиотек

# Вводные данные
A = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9], 
             [8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9]])

# Соседи
k = 3

# Тест
test_data = [3.3, 2.9]

# Визуализиция ввода даных
plt.figure()
plt.title("input data")
plt.scatter(A[:, 0], A[:, 1], marker = "o", s = 100, c = "black")

# Создание ближ. соседей с обучением
knn_model = NearestNeighbors(n_neighbors = k, algorithm = "auto").fit(A)
distances, indices = knn_model.kneighbors([test_data])

# Коорды ближ. Соседей
print("\nk Neighboring neighbors: ")
for rank, index in enumerate(indices[0][:k], start = 1):
    print(str(rank) + " is", A[index])

#визуализируция соседей
plt.title("K Nearest Neighbors")
plt.scatter(A[:, 0], A[:, 1], marker="o", s=100, c="k")
plt.scatter(A[indices][0][:][:, 0], A[indices][0][:][:, 1], marker = "o", s=250, facecolors = 'none', edgecolors='purple')
plt.scatter(test_data[0], test_data[1], marker = "x", c = "purple", s = 100)

plt.show()
