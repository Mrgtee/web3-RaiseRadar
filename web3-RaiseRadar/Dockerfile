FROM python:3.12-slim
WORKDIR /app

# 1. Copy the "shopping list" first
COPY requirements.txt .

# 2. Tell the server to install everything on that list
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of your code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
