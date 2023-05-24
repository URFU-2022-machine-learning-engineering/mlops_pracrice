from catboost.datasets import titanic


# Загрузка данных
train, test = titanic()

# обработка данных
# заполним данные о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
train['Sex'] = train['Sex'].apply(lambda x: 0 if x == 'male' else 1)
test['Sex'] = test['Sex'].apply(lambda x: 0 if x == 'male' else 1)
# в признаке "Возраст" много пропущенных (NaN) значений, заполним их средним значением возраста
train['Age'] = train['Age'].fillna(train.Age.mean())
test['Age'] = test['Age'].fillna(train.Age.mean())

# запишем созданные датасеты во внешние csv файлы
train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']].to_csv('data_train.csv', index=False)
test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].to_csv('data_test.csv', index=False)
