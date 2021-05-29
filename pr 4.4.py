from sklearn.datasets import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#отображание изображений цифр для проверки
def Image_display(i):
    plt.imshow(digit['images'][i],cmap = 'Greys_r')
    plt.show()
    
#загружаем набор данных MNIST
digit = load_digits()
digit_d = pd.DataFrame(digit['data'][0:1600]) #используем 1600 изображений для обучающего образца, оставшиеся 197 сохранены для тестирования

#создаем набор данных обучения и тестирования 
train_x = digit['data'][:1600]
train_y = digit['target'][:1600]

#предоставляем набор данных тестирования для классификаторов KNN
KNN = KNeighborsClassifier(20)
KNN.fit(train_x,train_y)

#создаем конструктор классификатора ближайшего соседа
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 20, p = 2, weights = 'uniform')

#создаем тестовый образец, указав любое произвольное число больше 1600
test = np.array(digit['data'][1725])
test1 = test.reshape(1,-1)
Image_display(1725)

#прогнозируем данные теста
print(KNN.predict(test1))

print(digit['target_names'])