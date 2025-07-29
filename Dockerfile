FROM python:3.11.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn
COPY api/ /app/api/
COPY output/ /app/output
WORKDIR /app/api
EXPOSE 8000
ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
