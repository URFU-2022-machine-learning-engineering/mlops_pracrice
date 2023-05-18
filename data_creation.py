import os
import random
import numpy as np

# Создаем папки "train" и "test", если они не существуют
if not os.path.exists("train"):
    os.makedirs("train")
if not os.path.exists("test"):
    os.makedirs("test")

# Генерируем данные и сохраняем их в папках "train" и "test"
for i in range(1, 6):
    # Генерируем случайные значения температуры в течение 30 дней
    temperatures = np.random.uniform(low=20, high=30, size=30)

    # Добавляем аномалии или шумы
    if i % 2 == 0:
        # Добавляем аномалии в четные наборы данных
        anomaly_indices = random.sample(range(30), 3)
        temperatures[anomaly_indices] += np.random.uniform(low=5, high=10, size=3)
    else:
        # Добавляем шумы в нечетные наборы данных
        temperatures += np.random.uniform(low=-2, high=2, size=30)

    # Разделение на train и test
    if i <= 3:
        folder = "train"
    else:
        folder = "test"

    # Сохраняем данные в файл
    filename = f"{folder}/data_{i}.txt"
    np.savetxt(filename, temperatures, delimiter=",")

    print(f"Создан файл {filename}")

print("Генерация данных завершена.")
