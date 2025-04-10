FROM python:3.12.10-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY README.md .
COPY styles.css .
COPY /ai-detector ./ai-detector
COPY /model-detector ./model-detector
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]