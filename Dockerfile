FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord : Docker met cette couche en cache si le fichier ne change pas
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ app/
COPY model/ model/

# Ne pas tourner en root dans le conteneur
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]