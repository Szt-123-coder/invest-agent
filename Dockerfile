FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY evals ./evals
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# 数据库放在 /app/data，挂一个卷就能在重启后保留：docker run -v invest-data:/app/data ...
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
