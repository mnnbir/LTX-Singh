FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -e .[inference-script]
RUN pip install gradio

EXPOSE 7860
CMD ["python", "app.py"]
