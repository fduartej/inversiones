# Imagen base
FROM python:3.10-slim

# Evitar input interactivo en pip
ENV DEBIAN_FRONTEND=noninteractive

# Crear y moverse a app
WORKDIR /app

# Copiar archivos
COPY requirements.txt .
COPY app.py .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
