FROM python:3.11-slim

WORKDIR /app

# Copiar todos los archivos al contenedor
COPY . .

# Instalar dependencias

# Instalar librería del sistema necesaria para LightGBM
RUN apt-get update && apt-get install -y libgomp1

RUN pip install --no-cache-dir -r requirements.txt


# Exponer el puerto para Railway
EXPOSE 8000

# Comando para ejecutar la API principal con Uvicorn
CMD ["uvicorn", "fase3_api:app", "--host", "0.0.0.0", "--port", "8000"]
