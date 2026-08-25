FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmagic1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

EXPOSE 5000

# timeout 300 согласован с proxy_read_timeout в nginx: выгрузка у верхней
# границы лимита (50 МБ / ~175k строк) обрабатывается дольше 120 с и раньше
# роняла воркер по таймауту, отдавая пользователю голый 502
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]
