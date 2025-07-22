# ==============================================================================
# FASE 1: PROCESAMIENTO DE MICROARRAYS GSE62872 - CÁNCER DE PRÓSTATA
# Script optimizado usando el enfoque específico del usuario
# ==============================================================================

# Instalar si es necesario
# BiocManager::install(c("GEOquery", "oligo", "limma", "pd.hugene.1.0.st.v1", 
#                        "hugene10sttranscriptcluster.db", "tidyverse"))

# Cargar librerías
library(GEOquery)
library(oligo)
library(limma)
library(tidyverse)
library(pd.hugene.1.0.st.v1)
library(hugene10sttranscriptcluster.db)

# Crear directorios
dir.create("data", showWarnings = FALSE)
dir.create("data/processed", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

cat("=== INICIANDO PROCESAMIENTO DE MICROARRAYS GSE62872 ===\n")

# ==============================================================================
# 1. DESCARGA Y CARGA DE DATOS GSE62872
# ==============================================================================

cat("=== DESCARGANDO DATASET GSE62872 ===\n")

# Descargar datos de GEO
gse <- getGEO("GSE62872", GSEMatrix = TRUE)
eset <- gse[[1]]

cat("✓ Dataset descargado exitosamente\n")

# ==============================================================================
# 2. EXTRACCIÓN DE DATOS DE EXPRESIÓN Y METADATOS
# ==============================================================================

cat("=== EXTRAYENDO DATOS DE EXPRESIÓN Y METADATOS ===\n")

# Expresión y metadatos
expr_data <- exprs(eset)
pheno_data <- pData(eset)

cat("Dimensiones originales:\n")
cat("- Genes:", nrow(expr_data), "\n")
cat("- Muestras:", ncol(expr_data), "\n")

# ==============================================================================
# 3. CLASIFICACIÓN DE MUESTRAS (TUMOR VS NORMAL)
# ==============================================================================

cat("=== CLASIFICANDO MUESTRAS ===\n")

# Usar título para extraer clase (tumor vs normal)
pheno_data$class <- case_when(
  str_detect(tolower(pheno_data$title), "tumor") ~ 1,
  str_detect(tolower(pheno_data$title), "normal") ~ 0,
  TRUE ~ NA_real_
)

# Mostrar algunos ejemplos de títulos y clasificaciones
cat("Ejemplos de clasificación:\n")
sample_titles <- head(pheno_data[, c("title", "class")], 10)
print(sample_titles)

# ==============================================================================
# 4. FILTRADO DE MUESTRAS CON CLASE DEFINIDA
# ==============================================================================

cat("=== FILTRANDO MUESTRAS ===\n")

# Filtrar muestras con clase definida
valid_samples <- !is.na(pheno_data$class)
expr_data <- expr_data[, valid_samples]
pheno_data <- pheno_data[valid_samples, ]

cat("Después del filtrado:\n")
cat("- Muestras válidas:", ncol(expr_data), "\n")
cat("- Distribución de clases:\n")
print(table(pheno_data$class))

# ==============================================================================
# 5. NORMALIZACIÓN Y TRANSFORMACIÓN LOG2
# ==============================================================================

cat("=== VERIFICANDO Y APLICANDO TRANSFORMACIÓN LOG2 ===\n")

# Comprobar si está en escala log2, y aplicar si no
if (max(expr_data, na.rm = TRUE) > 100) {
  cat("- Aplicando transformación log2...\n")
  expr_data <- log2(expr_data + 1)
} else {
  cat("- Los datos ya están en escala log2\n")
}

cat("Rango de valores después de normalización:", 
    round(range(expr_data, na.rm = TRUE), 2), "\n")

# ==============================================================================
# 6. ANÁLISIS DIFERENCIAL CON LIMMA
# ==============================================================================

cat("=== REALIZANDO ANÁLISIS DIFERENCIAL CON LIMMA ===\n")

# Diseño experimental para ANOVA con limma
design <- model.matrix(~ factor(pheno_data$class))
fit <- lmFit(expr_data, design)
fit <- eBayes(fit)

cat("✓ Análisis diferencial completado\n")

# ==============================================================================
# 7. SELECCIÓN DE GENES SIGNIFICATIVOS
# ==============================================================================

cat("=== SELECCIONANDO GENES SIGNIFICATIVOS ===\n")

# Obtener p-values y FDR
pvals <- fit$p.value[, 2]
fdr <- p.adjust(pvals, method = "fdr")

# Seleccionar genes significativos (FDR < 0.01)
selected_genes <- which(fdr < 0.01)

cat("Genes significativos (FDR < 0.01):", length(selected_genes), "\n")

# Si hay muy pocos genes, relajar el criterio
if (length(selected_genes) < 100) {
  cat("Pocos genes significativos, usando FDR < 0.05...\n")
  selected_genes <- which(fdr < 0.05)
  cat("Genes con FDR < 0.05:", length(selected_genes), "\n")
}

# Si aún hay pocos, tomar los top 1000
if (length(selected_genes) < 100) {
  cat("Usando top 1000 genes con menor FDR...\n")
  selected_genes <- order(fdr)[1:min(1000, length(fdr))]
  cat("Genes seleccionados:", length(selected_genes), "\n")
}

# ==============================================================================
# 8. PREPARACIÓN DE DATOS FINALES
# ==============================================================================

cat("=== PREPARANDO DATOS FINALES ===\n")

# Extraer expresión de genes seleccionados
expr_selected <- t(expr_data[selected_genes, ])
df_final <- as.data.frame(expr_selected)
df_final$class <- factor(pheno_data$class)

# Limpiar nombres de columnas
colnames(df_final) <- gsub("[^A-Za-z0-9_]", "_", colnames(df_final))

cat("Dimensiones finales:\n")
cat("- Muestras:", nrow(df_final), "\n")
cat("- Genes:", ncol(df_final) - 1, "\n")
cat("- Distribución de clases:\n")
print(table(df_final$class))

# ==============================================================================
# 9. GUARDADO DE RESULTADOS
# ==============================================================================

cat("=== GUARDANDO RESULTADOS ===\n")

# Guardar dataset principal
write.csv(df_final, "data/processed/expression_selected.csv", row.names = TRUE)

# Guardar información adicional de genes
genes_info <- data.frame(
  gene_id = rownames(expr_data)[selected_genes],
  p_value = pvals[selected_genes],
  fdr = fdr[selected_genes],
  log_fc = fit$coefficients[selected_genes, 2]
)

write.csv(genes_info, "results/genes_differential_analysis.csv", row.names = FALSE)

# Guardar información de muestras
sample_info <- data.frame(
  sample_id = rownames(pheno_data),
  title = pheno_data$title,
  class = pheno_data$class,
  class_label = ifelse(pheno_data$class == 1, "tumor", "normal")
)

write.csv(sample_info, "results/sample_information.csv", row.names = FALSE)

# ==============================================================================
# 10. REPORTE FINAL
# ==============================================================================

cat("\n=== REPORTE FINAL ===\n")
cat("✅ Dataset GSE62872 procesado exitosamente\n")
cat("✅ Transformación log2 aplicada\n")
cat("✅ Análisis diferencial con limma completado\n")
cat("✅ Genes significativos seleccionados\n")

cat("\nArchivos generados:\n")
cat("📁 data/processed/expression_selected.csv - Dataset para ML\n")
cat("📁 results/genes_differential_analysis.csv - Información de genes\n")
cat("📁 results/sample_information.csv - Información de muestras\n")

cat("\nEstadísticas finales:\n")
cat("- Total de muestras:", nrow(df_final), "\n")
cat("- Genes seleccionados:", ncol(df_final) - 1, "\n")
cat("- Muestras normales (clase 0):", sum(df_final$class == 0), "\n")
cat("- Muestras tumorales (clase 1):", sum(df_final$class == 1), "\n")

cat("\n✅ FASE 1 COMPLETADA EXITOSAMENTE\n")
cat("📋 Archivo principal guardado en:", file.path(getwd(), "data/processed/expression_selected.csv"), "\n")
cat("🔜 Siguiente paso: Ejecutar Fase 2 (Machine Learning en Python)\n")