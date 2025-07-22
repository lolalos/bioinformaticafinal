# ==============================================================================
# FASE 3: API FASTAPI PARA PREDICCIÓN DE CÁNCER DE PRÓSTATA
# ==============================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

# ==============================================================================
# CONFIGURACIÓN Y CARGA DE MODELOS
# ==============================================================================

app = FastAPI(
    title="Prostate Cancer Prediction API",
    description="API para predicción de cáncer de próstata usando datos de microarrays",
    version="1.0.0"
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
scaler = None
feature_selector = None
selected_features = None
model_metadata = None

def load_models():
    global model, scaler, feature_selector, selected_features, model_metadata
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_selector = joblib.load('models/feature_selector.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        model_metadata = joblib.load('models/model_metadata.pkl')
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        raise Exception(f"Error al cargar los modelos: {e}")

load_models()

# ==============================================================================
# MODELOS PYDANTIC PARA VALIDACIÓN
# ==============================================================================

class PredictionResponse(BaseModel):
    prediction: str
    probability_normal: float
    probability_tumor: float
    confidence: float
    timestamp: str
    model_used: str
    statistics: Dict[str, float] = None
    chart_url: str = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    features_count: int
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]
    statistics: Dict[str, Any]
    chart_url: str = None

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def validate_csv_data(df: pd.DataFrame) -> tuple:
    if df.empty:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    if len(df.columns) < len(selected_features):
        raise HTTPException(
            status_code=400, 
            detail=f"El CSV debe tener al menos {len(selected_features)} columnas de genes"
        )
    available_features = []
    missing_features = []
    for feature in selected_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    if len(missing_features) > 0:
        if len(missing_features) > len(selected_features) * 0.5:
            available_features = df.columns[:len(selected_features)].tolist()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan características importantes: {missing_features[:10]}..."
            )
    return available_features, df

def preprocess_data(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    try:
        X = df[features]
        if X.isnull().any().any():
            X = X.fillna(X.median())
        X_scaled = scaler.transform(X)
        X_selected = feature_selector.transform(X_scaled)
        return X_selected
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en el preprocesamiento: {str(e)}"
        )

def make_prediction(X_processed: np.ndarray) -> dict:
    try:
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0]
        result = {
            'prediction': 'tumor' if prediction == 1 else 'normal',
            'probability_normal': float(probabilities[0]),
            'probability_tumor': float(probabilities[1]),
            'confidence': float(max(probabilities)),
            'timestamp': datetime.now().isoformat(),
            'model_used': model_metadata['best_model_name']
        }
        # Estadísticas individuales
        result['statistics'] = {
            'confidence': result['confidence'],
            'probability_normal': result['probability_normal'],
            'probability_tumor': result['probability_tumor']
        }
        # Gráfico individual
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['Normal', 'Tumor'], [result['probability_normal'], result['probability_tumor']], color=['green', 'red'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilidad')
        ax.set_title('Probabilidades de Predicción')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        result['chart_url'] = f"data:image/png;base64,{img_base64}"
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )

def make_batch_chart(predictions: List[PredictionResponse]) -> str:
    # Gráfico de barras: cantidad de tumor vs normal
    labels = ['Normal', 'Tumor']
    counts = [sum(p.prediction == 'normal' for p in predictions), sum(p.prediction == 'tumor' for p in predictions)]
    confidences = [p.confidence for p in predictions]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].bar(labels, counts, color=['green', 'red'])
    axs[0].set_title('Conteo de Predicciones')
    axs[0].set_ylabel('Cantidad')
    axs[1].hist(confidences, bins=10, color='blue', alpha=0.7)
    axs[1].set_title('Distribución de Confianza')
    axs[1].set_xlabel('Confianza')
    axs[1].set_ylabel('Frecuencia')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# ==============================================================================
# ENDPOINTS DE LA API
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Proyecto Semestral | Bioinformática</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .gradient-bg {{
                background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%);
            }}
            .card {{
                box-shadow: 0 4px 24px rgba(37,99,235,0.08);
            }}
        </style>
    </head>
    <body class="bg-gray-100">
        <div class="max-w-4xl mx-auto mt-10">
            <div class="gradient-bg rounded-xl p-6 mb-8 text-white card">
                <h1 class="text-4xl font-extrabold mb-2 flex items-center gap-2">
                    <span>🎓</span> Proyecto Semestral
                </h1>
                <div class="flex flex-col md:flex-row md:gap-10">
                    <div class="mb-4 md:mb-0">
                        <h2 class="text-xl font-semibold">ASIGNATURA:</h2>
                        <p class="text-lg">Bioinformática</p>
                        <h2 class="text-xl font-semibold mt-2">Docente:</h2>
                        <p class="text-lg">María del Pilar Venegas Vergara</p>
                    </div>
                    <div>
                        <h2 class="text-xl font-semibold">Integrantes:</h2>
                        <ul class="list-disc ml-5 text-lg">
                            <li>Sanchez Palomino Dennis Oswaldo</li>
                            <li>Vitorino Marin Efrain</li>
                            <li>Villalobos Usca Anghelo Jhulino</li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="bg-white rounded-xl shadow-lg p-8 card">
                <h2 class="text-2xl font-bold text-blue-700 mb-4 flex items-center gap-2">
                    <span>🧬</span> Clasificación de enfermedades a partir de perfiles de expresión génica
                </h2>
                <div class="bg-blue-50 p-4 rounded mb-6">
                    <h3 class="text-lg font-semibold text-blue-800 mb-2">🎯 Objetivo</h3>
                    <p class="text-gray-700 text-base">
                        Entrenar un modelo de machine learning que, a partir de expresiones génicas extraídas de microarrays, clasifique si un paciente tiene o no una enfermedad (por ejemplo, cáncer de próstata).
                    </p>
                </div>
                <h2 class="text-xl font-semibold text-gray-800 mb-2">📤 Subir Archivo para Predicción</h2>
                <form id="uploadForm" class="flex flex-col gap-3 md:flex-row md:items-center" enctype="multipart/form-data">
                    <input type="file" id="fileInput" accept=".csv" class="block w-full md:w-auto border border-gray-300 rounded px-3 py-2" required>
                    <select id="modeSelect" class="block w-full md:w-auto border border-gray-300 rounded px-3 py-2">
                        <option value="single">Predicción Individual</option>
                        <option value="batch">Predicción por Lotes</option>
                    </select>
                    <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">Predecir</button>
                </form>
                <div class="text-xs text-gray-500 mt-2">
                    <ul class="list-disc ml-5">
                        <li>Sube uno o varios archivos CSV con datos de expresión génica</li>
                        <li>Cada fila representa una muestra</li>
                        <li>Las columnas deben ser los genes usados en el entrenamiento</li>
                    </ul>
                </div>
                <div class="mt-8">
                    <h3 class="text-lg font-semibold text-blue-800 mb-2">📊 Información del Modelo</h3>
                    <ul class="text-gray-700">
                        <li><b>Modelo:</b> {model_metadata['best_model_name']}</li>
                        <li><b>Características:</b> {model_metadata['selected_features_count']} genes seleccionados</li>
                        <li><b>Accuracy:</b> {model_metadata['metrics']['accuracy']:.4f}</li>
                        <li><b>F1-Score:</b> {model_metadata['metrics']['f1_score']:.4f}</li>
                    </ul>
                </div>
                <div id="result"></div>
                <div id="chartDiv" class="mt-4"></div>
                <div id="batchTable" class="mt-6"></div>
                <div id="batchChartDiv" class="mt-4"></div>
                <div id="downloadBtn" class="mt-4"></div>
            </div>
        </div>
        <footer class="mt-10 text-center text-gray-500 text-sm">
            <hr class="mb-4">
            <div>© {datetime.now().year} Proyecto Semestral Bioinformática | universidad nacional san antonio abad del cusco escuela profesional ing informatica y de sistemas</div>
            <div class="mt-4 flex justify-center gap-4">
                <a href="/docs" target="_blank" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">Acceder a API</a>
                <a href="https://github.com/lolalos/bioinformaticafinal" target="_blank" class="bg-gray-700 text-white px-4 py-2 rounded hover:bg-gray-800 transition">Acceder al código fuente</a>
            </div>
        </footer>
        <script>
        // Permitir múltiples archivos solo en modo batch
        const fileInput = document.getElementById('fileInput');
        const modeSelect = document.getElementById('modeSelect');
        modeSelect.addEventListener('change', function() {{
            if (modeSelect.value === 'batch') {{
                fileInput.setAttribute('multiple', 'multiple');
            }} else {{
                fileInput.removeAttribute('multiple');
            }}
        }});
        if (modeSelect.value === 'batch') {{
            fileInput.setAttribute('multiple', 'multiple');
        }} else {{
            fileInput.removeAttribute('multiple');
        }}
        function downloadCSV(data, filename) {{
            const csvRows = [];
            const headers = Object.keys(data[0]);
            csvRows.push(headers.join(','));
            for (const row of data) {{
                csvRows.push(headers.map(h => row[h]).join(','));
            }}
            const csvString = csvRows.join('\\n');
            const blob = new Blob([csvString], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('download', filename);
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }}
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {{
            e.preventDefault();
            const mode = modeSelect.value;
            const resultDiv = document.getElementById('result');
            const chartDiv = document.getElementById('chartDiv');
            const batchTableDiv = document.getElementById('batchTable');
            const batchChartDiv = document.getElementById('batchChartDiv');
            const downloadBtnDiv = document.getElementById('downloadBtn');
            resultDiv.innerHTML = "";
            chartDiv.innerHTML = "";
            batchTableDiv.innerHTML = "";
            batchChartDiv.innerHTML = "";
            downloadBtnDiv.innerHTML = "";
            if (!fileInput.files.length) {{
                resultDiv.innerHTML = '<div class="bg-red-100 text-red-700 p-3 rounded">Por favor selecciona al menos un archivo CSV</div>';
                return;
            }}
            const formData = new FormData();
            if (mode === "batch") {{
                for (let i = 0; i < fileInput.files.length; i++) {{
                    formData.append('files', fileInput.files[i]);
                }}
            }} else {{
                formData.append('file', fileInput.files[0]);
            }}
            let endpoint = mode === "batch" ? "/predict_batch" : "/predict";
            try {{
                resultDiv.innerHTML = '<div class="bg-blue-100 text-blue-700 p-3 rounded">Procesando...</div>';
                const response = await fetch(endpoint, {{
                    method: 'POST',
                    body: formData
                }});
                const result = await response.json();
                if (response.ok) {{
                    if (mode === "single") {{
                        resultDiv.innerHTML = `
                            <div class="bg-green-100 text-green-800 p-4 rounded">
                                <h3 class="font-bold text-lg mb-2">📋 Resultado de la Predicción</h3>
                                <ul class="space-y-1">
                                    <li><b>Predicción:</b> <span class="uppercase">${{result.prediction}}</span></li>
                                    <li><b>Probabilidad Normal:</b> ${{(result.probability_normal * 100).toFixed(2)}}%</li>
                                    <li><b>Probabilidad Tumor:</b> ${{(result.probability_tumor * 100).toFixed(2)}}%</li>
                                    <li><b>Confianza:</b> ${{(result.confidence * 100).toFixed(2)}}%</li>
                                    <li><b>Modelo usado:</b> ${{result.model_used}}</li>
                                    <li><b>Timestamp:</b> ${{result.timestamp}}</li>
                                </ul>
                                <div class="mt-3">
                                    <b>Estadísticas:</b>
                                    <ul>
                                        <li>Confianza: ${{(result.statistics.confidence*100).toFixed(2)}}%</li>
                                        <li>Probabilidad Normal: ${{(result.statistics.probability_normal*100).toFixed(2)}}%</li>
                                        <li>Probabilidad Tumor: ${{(result.statistics.probability_tumor*100).toFixed(2)}}%</li>
                                    </ul>
                                </div>
                            </div>
                        `;
                        chartDiv.innerHTML = `<img src="${{result.chart_url}}" alt="Gráfico de probabilidades" class="rounded shadow mt-2" style="max-width:300px;">`;
                    }} else {{
                        let tableHtml = `<div class="bg-green-50 p-4 rounded">
                            <h3 class="font-bold text-lg mb-2">📋 Resultados de Predicción por Lotes</h3>
                            <div class="overflow-x-auto">
                            <table class="min-w-full text-sm border border-gray-300 rounded">
                                <thead>
                                    <tr class="bg-blue-100">
                                        <th class="px-2 py-1 border">#</th>
                                        <th class="px-2 py-1 border">Predicción</th>
                                        <th class="px-2 py-1 border">Prob. Normal</th>
                                        <th class="px-2 py-1 border">Prob. Tumor</th>
                                        <th class="px-2 py-1 border">Confianza</th>
                                        <th class="px-2 py-1 border">Timestamp</th>
                                    </tr>
                                </thead>
                                <tbody>`;
                        result.predictions.forEach((p, idx) => {{
                            tableHtml += `<tr class="${{p.prediction==='tumor'?'bg-red-50':'bg-green-50'}}">
                                <td class="px-2 py-1 border">${{idx+1}}</td>
                                <td class="px-2 py-1 border uppercase">${{p.prediction}}</td>
                                <td class="px-2 py-1 border">${{(p.probability_normal*100).toFixed(2)}}%</td>
                                <td class="px-2 py-1 border">${{(p.probability_tumor*100).toFixed(2)}}%</td>
                                <td class="px-2 py-1 border">${{(p.confidence*100).toFixed(2)}}%</td>
                                <td class="px-2 py-1 border">${{p.timestamp}}</td>
                            </tr>`;
                        }});
                        tableHtml += `</tbody></table></div>
                            <div class="mt-3 text-gray-700">
                                <b>Total muestras:</b> ${{result.summary.total_samples}} |
                                <b>Tumor:</b> ${{result.summary.tumor_predictions}} |
                                <b>Normal:</b> ${{result.summary.normal_predictions}} |
                                <b>% Tumor:</b> ${{result.summary.tumor_percentage.toFixed(2)}}% |
                                <b>Confianza Promedio:</b> ${{result.summary.average_confidence.toFixed(2)}}
                            </div>
                            <div class="mt-3 text-gray-700">
                                <b>Estadísticas:</b> <br>
                                <ul>
                                    <li><b>Confianza Máxima:</b> ${{result.statistics.max_confidence.toFixed(2)}}</li>
                                    <li><b>Confianza Mínima:</b> ${{result.statistics.min_confidence.toFixed(2)}}</li>
                                    <li><b>Desviación estándar confianza:</b> ${{result.statistics.std_confidence.toFixed(2)}}</li>
                                </ul>
                            </div>
                        </div>`;
                        batchTableDiv.innerHTML = tableHtml;
                        batchChartDiv.innerHTML = `<img src="${{result.chart_url}}" alt="Gráfico de estadísticas" class="rounded shadow mt-2" style="max-width:500px;">`;
                        downloadBtnDiv.innerHTML = `<button id="downloadCsvBtn" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">Descargar resultados CSV</button>`;
                        document.getElementById('downloadCsvBtn').onclick = () => downloadCSV(result.predictions, "predicciones.csv");
                        resultDiv.innerHTML = "";
                        chartDiv.innerHTML = "";
                    }}
                }} else {{
                    resultDiv.innerHTML = `<div class="bg-red-100 text-red-700 p-3 rounded">Error: ${{result.detail}}</div>`;
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<div class="bg-red-100 text-red-700 p-3 rounded">Error de conexión: ${{error.message}}</div>`;
            }}
        }});
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_metadata['best_model_name'] if model_metadata else "Unknown",
        features_count=len(selected_features) if selected_features else 0,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        features, df = validate_csv_data(df)
        sample_df = df.iloc[[0]]
        result = make_prediction(preprocess_data(sample_df, features))
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    all_predictions = []
    tumor_count = 0
    normal_count = 0
    confidences = []
    total_samples = 0
    try:
        for file in files:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail=f"Solo se aceptan archivos CSV ({file.filename})")
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            features, df = validate_csv_data(df)
            X_processed = preprocess_data(df, features)
            for i in range(len(X_processed)):
                sample_processed = X_processed[i:i+1]
                result = make_prediction(sample_processed)
                all_predictions.append(PredictionResponse(**result))
                confidences.append(result['confidence'])
                if result['prediction'] == 'tumor':
                    tumor_count += 1
                else:
                    normal_count += 1
            total_samples += len(X_processed)
        summary = {
            'total_samples': len(all_predictions),
            'tumor_predictions': tumor_count,
            'normal_predictions': normal_count,
            'tumor_percentage': (tumor_count / len(all_predictions)) * 100 if len(all_predictions) > 0 else 0,
            'average_confidence': np.mean(confidences) if confidences else 0
        }
        statistics = {
            'max_confidence': np.max(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0
        }
        chart_url = make_batch_chart(all_predictions)
        return BatchPredictionResponse(
            predictions=all_predictions,
            summary=summary,
            statistics=statistics,
            chart_url=chart_url
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivos: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    if not model_metadata:
        raise HTTPException(status_code=500, detail="Metadatos del modelo no disponibles")
    return {
        "model_name": model_metadata['best_model_name'],
        "model_parameters": model_metadata['best_model_params'],
        "metrics": model_metadata['metrics'],
        "selected_features_count": model_metadata['selected_features_count'],
        "training_info": {
            "total_samples": model_metadata['total_samples'],
            "train_samples": model_metadata['train_samples'],
            "test_samples": model_metadata['test_samples']
        }
    }

@app.get("/features")
async def get_selected_features():
    if not selected_features:
        raise HTTPException(status_code=500, detail="Lista de características no disponible")
    return {
        "selected_features": selected_features,
        "count": len(selected_features)
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno del servidor: {str(exc)}"}
    )

if __name__ == "__main__":
    print("="*60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE CÁNCER DE PRÓSTATA")
    print("="*60)
    if model is None:
        print("❌ Error: No se pudieron cargar los modelos")
        print("Asegúrate de haber ejecutado las Fases 1 y 2 primero")
        exit(1)
    print(f"✅ Modelo cargado: {model_metadata['best_model_name']}")
    print(f"✅ Características: {len(selected_features)}")
    print("\n📡 Endpoints disponibles:")
    print("• GET  /          - Interfaz web")
    print("• GET  /health    - Estado de la API")
    print("• POST /predict   - Predicción individual")
    print("• POST /predict_batch - Predicción por lotes")
    print("• GET  /docs      - Documentación Swagger")
    print("\n🌐 Iniciando servidor en: http://localhost:8000")
    print("📚 Documentación en: http://localhost:8000/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
