FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt

COPY main.py /app/main.py
ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000
# Using plain python here (deps already synced via uv). You could also: ["uv","run","uvicorn",...]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
