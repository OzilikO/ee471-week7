# 1. Base image with lightweight Python
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# 4. Copy model weights and application files
COPY baseline_model.pth .
COPY optimized_model.pth .
COPY predict.py .
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/

# 5. Expose web server port
EXPOSE 8080

# 6. Default environment variables
ENV COG_URL="http://localhost:5000/predictions"
ENV PORT=8080

# 7. Start the Flask web application
CMD ["python", "app.py"]
