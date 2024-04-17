import pandas as pd

df = pd.read_csv('datasets/mean_age_passenger.csv', delimiter =',')
one_hot_encoded = pd.get_dummies(df['Sex'])
df = pd.concat([df, one_hot_encoded], axis=1)
df.to_csv('one_hot_encoding_passenger.csv', index = False)
