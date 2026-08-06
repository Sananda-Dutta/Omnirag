# Phase 1: minimal single-stage image, just enough to run the skeleton.
# Phase 18 will convert this to a proper multi-stage build (smaller final
# image, non-root user, layer caching tuned for requirements.txt vs app code).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
