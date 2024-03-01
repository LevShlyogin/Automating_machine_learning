from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_creation import df_se

# Определяем X и y, нам нужно предсказать цену
y = df_se["Price"]
X = df_se.drop(["Price"], axis = 1)

# Делим данные на тестовую и тренировочную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Сохраняем тренировочные и тестовые выборки в CSV файлы
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

train_data.to_csv('train/train_data.csv', index=False)
test_data.to_csv('test/test_data.csv', index=False)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)