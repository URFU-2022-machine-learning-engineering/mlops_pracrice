import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

# прочитаем из csv файла подготовленный датасет для обучения
data_train = pd.read_csv('data_train.csv')
X_train = data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].values
y_train = data_train['Survived'].values

# загрузим модель машинного обучения

model = LogisticRegression(max_iter=100_000).fit(X_train, y_train)

# сохраним обученную модель

pickle.dump(model, open('model.pkl', 'wb'))
