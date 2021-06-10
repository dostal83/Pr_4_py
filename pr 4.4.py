from sklearn.datasets import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier # Импорт библиотек

def Image_display(i):
    plt.imshow(digit['images'][i],cmap = 'Greys_r')
    plt.show()
    
# Загрузка данных
digit = load_digits()
digit_d = pd.DataFrame(digit['data'][0:1600]) #используем 1600 изображений для обучающего образца, оставшиеся 197 сохранены для тестирования

# Набор данный обучения/тест
train_x = digit['data'][:1600]
train_y = digit['target'][:1600]

# Набор данны тест KNN
KNN = KNeighborsClassifier(20)
KNN.fit(train_x,train_y)

# Классификатор ближ. соседей
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 20, p = 2, weights = 'uniform')

# Образцовый тест > 1600
test = np.array(digit['data'][1725])
test1 = test.reshape(1,-1)
Image_display(1725)

# Прогноз 
print(KNN.predict(test1))

print(digit['target_names'])
