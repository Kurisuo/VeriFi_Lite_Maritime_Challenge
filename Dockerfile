FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080
LABEL org.opencontainers.image.title="VeriFi-Lite" \
      org.opencontainers.image.revision="2026-07-14-r2"

CMD ["python", "main.py"]
