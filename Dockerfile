FROM python:3.11-slim

WORKDIR /app

# Copiar todos los archivos al contenedor
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar tu script principal
CMD ["python", "main.py"]
