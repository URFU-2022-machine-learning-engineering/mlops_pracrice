import pickle
import pandas as pd

loaded_model = pickle.load(open('model.pkl', 'rb'))

# прочитаем из csv файла подготовленный датасет для обучения
data_test = pd.read_csv('data_test.csv')
X_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].values

# сделаем предсказание для первого пассажира из тестовой выборки
print(loaded_model.predict(X_test[0:1]))
