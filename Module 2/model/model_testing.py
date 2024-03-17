import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import json

# Загрузка данных для тестирования
test_data = pd.read_csv('dataset/test/test_data_scaled.csv')

# Разделение предикторов и целевой переменной
X_test = test_data.drop(["Price"], axis=1)
y_test = test_data["Price"]

# Загрузка обученной модели
model = joblib.load('trained_model.pkl')

# Предсказание на тестовых данных
predictions = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, predictions)
print(f"Среднеквадратичная ошибка в тестовых данных: {mse}")

with open("testing_mse.json", "w", encoding="utf-8") as json_file:
    json.dump(mse, json_file, ensure_ascii=False)