import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Загрузка данных из папки "train"
train_data = []

for filename in os.listdir("train"):
    if filename.endswith(".txt"):
        filepath = os.path.join("train", filename)
        data = np.loadtxt(filepath, delimiter=",")
        train_data.append(data)

# Объединение всех обучающих примеров в один массив
X_train = np.concatenate(train_data)
y_train = np.arange(1, X_train.shape[0] + 1)  # Примерная зависимость для иллюстрации

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели в файл
filename = "model.pkl"
with open(filename, "wb") as file:
    pickle.dump(model, file)

print(f"Модель сохранена в файл {filename}.")
