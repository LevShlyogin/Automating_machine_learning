import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загружаем данные для предобработки
train_data = pd.read_csv('dataset/train/train_data.csv')
test_data = pd.read_csv('dataset/test/test_data.csv')

# Разделяем предикторы и целевую переменную
X_train = train_data.drop(["Price"], axis=1)
y_train = train_data["Price"]
X_test = test_data.drop(["Price"], axis=1)
y_test = test_data["Price"]

# Производим стандартизацию данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Сохраняем предобработанные данные
train_data_scaled = pd.concat([pd.DataFrame(X_train_scaled), y_train], axis=1)
test_data_scaled = pd.concat([pd.DataFrame(X_test_scaled), y_test], axis=1)

train_data_scaled.to_csv('dataset/train/train_data_scaled.csv', index=False)
test_data_scaled.to_csv('dataset/test/test_data_scaled.csv', index=False)