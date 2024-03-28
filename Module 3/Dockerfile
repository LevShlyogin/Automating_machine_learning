FROM python:3.9-slim
WORKDIR /app
EXPOSE 8501
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt

COPY . .

CMD ["streamlit", "run", "main.py"]
