FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080
LABEL org.opencontainers.image.title="VeriFi-Lite"

CMD ["python", "main.py"]
