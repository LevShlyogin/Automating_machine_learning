#!/bin/bash
echo "Устанавливаем переменные окружения Kaggle"
export KAGGLE_USERNAME="levshlyogin"
export KAGGLE_KEY="a34c45dc721891d278828abf7850aac4"

echo "Устанавливаем библиотеки и зависимости"
pip install -r requirements.txt

echo "Скачиваем датасет"
kaggle datasets download -d shivamkushwah0411/quicker-car-cleaned-dataset

echo "Создаём папку для сохранения данных"
mkdir -p dataset

echo "Извлекаем файл датасета в папку dataset"
unzip -o quicker-car-cleaned-dataset.zip -d dataset/quicker-car-cleaned-dataset

echo "Запускаем Python-скрипт для получения данных"
python3 data_creation.py
echo "data_creation.py Done!"
sleep 5

echo "Выполняем предобработку данных"
python3 model_preparation.py
echo "model_preparation.py Done!"
sleep 5

echo "Создаем и обучаем модель машинного обучения на построенных данных из папки train"
python3 model_preprocessing.py
echo "model_preprocessing.py Done!"
sleep 5

echo "Проверяем модель машинного обучения на построенных данных из папки test"
python3 model_testing.py
echo "model_testing.py Done!"
sleep 5