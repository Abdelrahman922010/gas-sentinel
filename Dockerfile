FROM python:3.11-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Pre-train the calibration & IsolationForest models at image build time
RUN python train_model.py --out models/gas_sentinel_ai.joblib && chmod -R a+rX /srv

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --ws-ping-interval 15 --ws-ping-timeout 20"]
