# ==============================================================================
# FASE 2: MACHINE LEARNING PIPELINE - CÁNCER DE PRÓSTATA
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== INICIANDO PIPELINE DE MACHINE LEARNING ===\n")

# ==============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ==============================================================================

print("=== CARGANDO DATOS ===")

# Cargar datos procesados desde R
try:
    df = pd.read_csv('data/processed/expression_selected.csv', index_col=0)
    print("✅ Datos cargados exitosamente")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo expression_selected.csv")
    print("Asegúrate de haber ejecutado la Fase 1 (script de R) primero")
    exit()

print(f"Dimensiones del dataset: {df.shape}")
print(f"Distribución de clases:\n{df['class'].value_counts()}")

# Verificar datos faltantes
print(f"Valores faltantes: {df.isnull().sum().sum()}")

# ==============================================================================
# 2. PREPARACIÓN DE DATOS
# ==============================================================================

print("\n=== PREPARANDO DATOS ===")

# Separar features y target
X = df.drop('class', axis=1)
y = df['class'].astype(int)

print(f"Features: {X.shape[1]}")
print(f"Muestras: {X.shape[0]}")

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Conjunto de entrenamiento: {X_train.shape}")
print(f"Conjunto de prueba: {X_test.shape}")

# ==============================================================================
# 3. ESCALADO DE DATOS
# ==============================================================================

print("\n=== ESCALANDO DATOS ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Escalado completado")

# ==============================================================================
# 4. REDUCCIÓN DE DIMENSIONALIDAD CON L1 (LASSO)
# ==============================================================================

print("\n=== REDUCCIÓN DE DIMENSIONALIDAD CON L1 ===")

# Usar Lasso para selección de características
lasso_selector = SelectFromModel(
    Lasso(alpha=0.01, random_state=42),
    threshold='median'
)

X_train_selected = lasso_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = lasso_selector.transform(X_test_scaled)

selected_features = X.columns[lasso_selector.get_support()]
print(f"Características seleccionadas: {len(selected_features)}")
print(f"Reducción: {X.shape[1]} → {X_train_selected.shape[1]}")

# ==============================================================================
# 5. DEFINICIÓN DE MODELOS
# ==============================================================================

print("\n=== DEFINIENDO MODELOS ===")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=1000),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbosity=-1)
}

print(f"Modelos definidos: {list(models.keys())}")

# ==============================================================================
# 6. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ==============================================================================

print("\n=== ENTRENANDO Y EVALUANDO MODELOS ===")

results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"\nEntrenando {name}...")
    
    # Entrenar modelo
    model.fit(X_train_selected, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_selected)
    
    # Probabilidades (para ROC)
    try:
        y_prob = model.predict_proba(X_test_selected)[:, 1]
    except:
        y_prob = model.decision_function(X_test_selected)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Almacenar resultados
    results[name] = {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1,
        'model': model
    }
    
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# ==============================================================================
# 7. IDENTIFICACIÓN DEL MEJOR MODELO
# ==============================================================================

print("\n=== IDENTIFICANDO MEJOR MODELO ===")

# Crear DataFrame con resultados
results_df = pd.DataFrame(results).T
print("\nResultados completos:")
print(results_df.round(4))

# Seleccionar mejor modelo basado en F1-Score
best_model_name = results_df['f1_score'].idxmax()
best_model = results[best_model_name]['model']

print(f"\n🏆 Mejor modelo: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")

# ==============================================================================
# 8. VISUALIZACIONES
# ==============================================================================

print("\n=== GENERANDO VISUALIZACIONES ===")

# Crear directorio para visualizaciones
import os
os.makedirs('results/visualizations', exist_ok=True)

# 1. Comparación de métricas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ['accuracy', 'recall', 'f1_score']
for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in models.keys()]
    bars = axes[i].bar(models.keys(), values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14)
    axes[i].set_ylabel('Score')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for j, v in enumerate(values):
        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Curvas ROC
plt.figure(figsize=(12, 8))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('results/visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Matriz de confusión para el mejor modelo
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Tumor'], 
            yticklabels=['Normal', 'Tumor'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Importancia de genes (para modelos que lo soporten)
if hasattr(best_model, 'feature_importances_'):
    # Para modelos tree-based
    importances = best_model.feature_importances_
    feature_names = selected_features
elif hasattr(best_model, 'coef_'):
    # Para modelos lineales
    importances = np.abs(best_model.coef_[0])
    feature_names = selected_features
else:
    importances = None

if importances is not None:
    # Tomar top 20 características más importantes
    top_indices = np.argsort(importances)[-20:]
    top_importances = importances[top_indices]
    top_features = feature_names[top_indices]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top 20 Gene Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Heatmap de expresión de genes más importantes
if importances is not None:
    top_10_genes = feature_names[np.argsort(importances)[-10:]]
    heatmap_data = X_test[top_10_genes]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=top_10_genes)
    plt.title('Heatmap of Top 10 Most Important Genes')
    plt.xlabel('Samples')
    plt.ylabel('Genes')
    plt.tight_layout()
    plt.savefig('results/visualizations/gene_expression_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

print("✅ Visualizaciones guardadas en results/visualizations/")

# ==============================================================================
# 9. PRUEBAS DE PREDICCIÓN CON 10 MUESTRAS ALEATORIAS
# ==============================================================================

print("\n=== REALIZANDO PRUEBAS DE PREDICCIÓN ===")

# Seleccionar 10 muestras aleatorias del conjunto de prueba
np.random.seed(42)
random_indices = np.random.choice(len(X_test), 10, replace=False)

print("Pruebas de predicción con 10 muestras aleatorias:")
print("-" * 60)

for i, idx in enumerate(random_indices):
    sample = X_test_selected[idx:idx+1]
    true_label = y_test.iloc[idx]
    predicted_label = best_model.predict(sample)[0]
    probability = best_model.predict_proba(sample)[0]
    
    print(f"Muestra {i+1}:")
    print(f"  Verdadero: {'Tumor' if true_label == 1 else 'Normal'}")
    print(f"  Predicho: {'Tumor' if predicted_label == 1 else 'Normal'}")
    print(f"  Probabilidades: Normal={probability[0]:.3f}, Tumor={probability[1]:.3f}")
    print(f"  {'✅ Correcto' if true_label == predicted_label else '❌ Incorrecto'}")
    print()

# ==============================================================================
# 10. GUARDADO DE MODELOS Y ARCHIVOS
# ==============================================================================

print("=== GUARDANDO MODELOS Y ARCHIVOS ===")

# Crear directorio para modelos
os.makedirs('models', exist_ok=True)

# Guardar el mejor modelo
joblib.dump(best_model, 'models/best_model.pkl')
print(f"✅ Mejor modelo guardado: {best_model_name}")

# Guardar scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler guardado")

# Guardar selector de características
joblib.dump(lasso_selector, 'models/feature_selector.pkl')
print("✅ Selector de características guardado")

# Guardar nombres de características seleccionadas
selected_features_list = selected_features.tolist()
joblib.dump(selected_features_list, 'models/selected_features.pkl')
print("✅ Lista de características seleccionadas guardada")

# Guardar metadatos del modelo
model_metadata = {
    'best_model_name': best_model_name,
    'best_model_params': best_model.get_params(),
    'metrics': results[best_model_name],
    'selected_features_count': len(selected_features),
    'total_samples': len(X),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

joblib.dump(model_metadata, 'models/model_metadata.pkl')
print("✅ Metadatos del modelo guardados")

# Guardar resultados en CSV
results_df.to_csv('results/model_results.csv')
print("✅ Resultados guardados en CSV")

# ==============================================================================
# 11. REPORTE FINAL
# ==============================================================================

print("\n" + "="*60)
print("🎉 FASE 2 COMPLETADA EXITOSAMENTE")
print("="*60)

print(f"\n📊 RESUMEN DE RESULTADOS:")
print(f"• Mejor modelo: {best_model_name}")
print(f"• Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"• Recall: {results[best_model_name]['recall']:.4f}")
print(f"• F1-Score: {results[best_model_name]['f1_score']:.4f}")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"• models/best_model.pkl - Mejor modelo entrenado")
print(f"• models/scaler.pkl - Escalador de datos")
print(f"• models/feature_selector.pkl - Selector de características")
print(f"• models/selected_features.pkl - Lista de genes seleccionados")
print(f"• models/model_metadata.pkl - Metadatos del modelo")
print(f"• results/model_results.csv - Resultados comparativos")

print(f"\n🖼️ VISUALIZACIONES:")
print(f"• results/visualizations/model_comparison.png")
print(f"• results/visualizations/roc_curves.png")
print(f"• results/visualizations/confusion_matrix.png")
print(f"• results/visualizations/feature_importance.png")
print(f"• results/visualizations/gene_expression_heatmap.png")

print(f"\n🔜 SIGUIENTE PASO: Fase 3 - Crear API con FastAPI")
print("="*60)