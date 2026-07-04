FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .[full]

COPY . .

EXPOSE 5000

ENV FLASK_APP=sma_app.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "sma_app.py"]
