import pandas as pd

df = pd.read_csv('passenger.csv')
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

df.to_csv('mean_age_passenger.csv', index=False)
