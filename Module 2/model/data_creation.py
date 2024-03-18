import pandas as pd
import os
from sklearn.model_selection import train_test_split

data = pd.read_csv('Module 2/model/dataset/quicker-car-cleaned-dataset/cleaned_car.csv', delimiter = ',', index_col=0)

# Смотрим на типы данных
cat_columns = []
num_columns = []

for column_name in data.columns:
    if (data[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]

print('Категориальные данные:\t ',cat_columns, '\n Число столблцов = ',len(cat_columns))

print('Числовые данные:\t ',  num_columns, '\n Число столблцов = ',len(num_columns))

# Перекодируем значения в колонке Fuel_type
data.fuel_type=data.fuel_type.replace({'Petrol':0,'Diesel':1,'LPG':2})

# Приводим категориальные признаки к численным
df_se = data.copy()
df_se[cat_columns] = df_se[cat_columns].astype('category')
for _, column_name in enumerate(cat_columns):
    df_se[column_name] = df_se[column_name].cat.codes

# Определяем X и y, нам нужно предсказать цену
y = df_se["Price"]
X = df_se.drop(["Price"], axis = 1)

# Делим данные на тестовую и тренировочную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Сохраняем тренировочные и тестовые выборки в CSV файлы
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Создаём папку для сохранения данных
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

train_data.to_csv('dataset/train/train_data.csv', index=False)
test_data.to_csv('dataset/test/test_data.csv', index=False)
