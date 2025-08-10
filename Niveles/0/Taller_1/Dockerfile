# Usar Python 3.11 slim como imagen base optimizada para aplicaciones científicas
FROM python:3.11-slim

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para las librerías científicas
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python primero (para cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar estructura completa del proyecto
COPY src/ ./src/
COPY api/ ./api/
COPY train_model.py .

# Crear directorio para modelos si no existe
RUN mkdir -p models

# Entrenar modelo durante construcción del contenedor
# Esto asegura que el contenedor incluya un modelo fresco entrenado
RUN python train_model.py

# Exponer puerto 8989 según especificaciones del taller
EXPOSE 8989

# Comando para ejecutar el API usando módulo Python
CMD ["python", "-m", "api.main"]
