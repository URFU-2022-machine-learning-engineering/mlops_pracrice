import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Загрузка данных из папок "train" и "test"
train_data = []
test_data = []

for filename in os.listdir("train"):
    if filename.endswith(".txt"):
        filepath = os.path.join("train", filename)
        data = np.loadtxt(filepath, delimiter=",")
        train_data.append(data)

for filename in os.listdir("test"):
    if filename.endswith(".txt"):
        filepath = os.path.join("test", filename)
        data = np.loadtxt(filepath, delimiter=",")
        test_data.append(data)

# Преобразование данных с использованием StandardScaler
scaler = StandardScaler()

# Объединяем обучающую и тестовую выборки для подгонки scaler
combined_data = np.concatenate(train_data + test_data).reshape(-1, 1)
scaler.fit(combined_data)

# Преобразование обучающей выборки
for i, data in enumerate(train_data):
    train_data[i] = scaler.transform(data.reshape(-1, 1))

# Преобразование тестовой выборки
for i, data in enumerate(test_data):
    test_data[i] = scaler.transform(data.reshape(-1, 1))

# Сохранение преобразованных данных
for i, data in enumerate(train_data):
    filename = f"train/data_{i+1}.txt"
    np.savetxt(filename, data, delimiter=",")

for i, data in enumerate(test_data):
    filename = f"test/data_{i+1}.txt"
    np.savetxt(filename, data, delimiter=",")

print("Предобработка данных завершена.")
