FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch to minimize container image footprint
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    fastapi uvicorn pillow python-multipart

# Copy service script and weights
COPY app.py vision_model.pth ./

EXPOSE 8000

# Start server listening across all interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
