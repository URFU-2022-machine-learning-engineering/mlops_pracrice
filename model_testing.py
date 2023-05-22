import os
import pickle
import numpy as np

# Load the model from file
model_filename = "model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Load the test data from the "test" folder
test_data = []

for filename in os.listdir("test"):
    if filename.endswith(".txt"):
        filepath = os.path.join("test", filename)
        data = np.loadtxt(filepath, delimiter=",")
        test_data.append(data)

# Convert the test data to a numpy array
test_data = np.array(test_data)

# Predict labels for the test data
predicted_labels = model.predict(test_data[:, 0].reshape(-1, 1))

# Calculate accuracy
accuracy = model.score(test_data[:, 0].reshape(-1, 1), predicted_labels)

print(f"Model test accuracy is: {accuracy:.3f}")

# Generate ground truth labels for the test data
ground_truth_labels = model.predict(test_data[:, 0].reshape(-1, 1))

# Save ground truth labels to file
ground_truth_folder = "ground_truth"
os.makedirs(ground_truth_folder, exist_ok=True)

for i, label in enumerate(ground_truth_labels):
    filename = f"ground_truth_label_{i+1}.txt"
    filepath = os.path.join(ground_truth_folder, filename)
    np.savetxt(filepath, label.reshape(1,), delimiter=",")

print("Ground truth labels for the test data have been created.")
