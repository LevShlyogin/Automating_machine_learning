import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Пример модели
import joblib  # Для сохранения модели

# Загрузка данных для обучения
train_data = pd.read_csv('dataset/train/train_data_scaled.csv')

# Разделение предикторов и целевой переменной
X_train = train_data.drop(["Price"], axis=1)
y_train = train_data["Price"]

# Создание и обучение модели
model = RandomForestRegressor()  # Пример модели
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'trained_model.pkl')