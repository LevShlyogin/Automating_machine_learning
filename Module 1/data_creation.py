import pandas as pd


data = pd.read_csv("cleaned_car.csv", delimiter = ',', index_col=0)

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

