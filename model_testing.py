import os
import pickle
import numpy as np

# Загрузка модели из файла
model_filename = "model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Загрузка данных из папки "test"
test_data = []

for filename in os.listdir("test"):
    if filename.endswith(".txt"):
        filepath = os.path.join("test", filename)
        data = np.loadtxt(filepath, delimiter=",")
        test_data.append(data)

# Проверка модели на данных из папки "test"
for i, data in enumerate(test_data):
    predicted = model.predict(data.reshape(-1, 1))
    print(f"Для файла test/data_{i + 1}.txt предсказанное значение: {predicted[0]}")

# Convert the test data to a numpy array
test_data = np.array(test_data)

# Predict labels for the test data
predicted_labels = model.predict(test_data[:, 0].reshape(-1, 1))

# Calculate accuracy
accuracy = model.score(test_data[:, 0].reshape(-1, 1), predicted_labels)

print(f"Model test accuracy is: {accuracy:.3f}")
