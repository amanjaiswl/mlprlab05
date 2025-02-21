FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install numpy pandas matplotlib scikit-learn wandb opencv-python-headless
CMD ["python", "main.py"]

