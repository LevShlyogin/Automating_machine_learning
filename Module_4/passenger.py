import os
import pandas as pd


def start():
    df = pd.read_csv('titanic.csv')
    print(df.head())
    ndf = df.loc[:,['Pclass', 'Sex', 'Age']]
    ndf.to_csv('passenger.csv', index=False)


if __name__ == '__main__':
    start()
