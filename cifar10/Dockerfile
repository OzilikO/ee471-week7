FROM python:3.11-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY baseline_model.pth .
COPY optimized_model.pth .
COPY predict.py .
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8081

ENV COG_URL="http://localhost:5001/predictions"
ENV PORT=8081

CMD ["python", "app.py"]
