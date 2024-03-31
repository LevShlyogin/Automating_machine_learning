# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем зависимости
RUN pip install --upgrade pip \
    && pip install streamlit transformers

# Устанавливаем зависимости для работы с изображениями
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Копируем код приложения в контейнер
COPY . /app
WORKDIR /app

# Устанавливаем дополнительные зависимости
RUN pip install -r requirements.txt

# Загружаем модель и процессор
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large'); \
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')"

# Указываем порт, который будет прослушиваться приложением
EXPOSE 8501

# Команда для запуска Streamlit приложения
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]