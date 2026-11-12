import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, LogisticRegression, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# Configuración del directorio de salida local
OUT_DIR = "/app/graficas_fraude"
os.makedirs(OUT_DIR, exist_ok=True)

# Rutas de salida para los gráficos locales
PLOT_COMPARISON_PATH = os.path.join(OUT_DIR, "modelo_comparacion_metricas.png")
PLOT_ROC_PATH = os.path.join(OUT_DIR, "modelo_curvas_roc.png")
PLOT_IMPORTANCE_PATH = os.path.join(OUT_DIR, "modelo_importancia_variables.png")

print("Iniciando SparkSession para entrenamiento y comparación de modelos (Modo Distribuido)...")
spark = SparkSession.builder \
    .appName("Fraude_Modelos_Comparacion") \
    .getOrCreate()

# Cargar Data Vectorizada desde HDFS
print("Cargando dataset vectorizado desde Hadoop HDFS...")
data = spark.read.parquet("hdfs://namenode:9000/user/hadoop/fraud/fraude_ml_ready.parquet")

# 1. División Aleatoria Distribuida con Semilla (Sección 10)
print("Dividiendo en Entrenamiento (80%) y Prueba (20%)...")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
train_data.cache()
test_data.cache()

print(f"Registros en Train: {train_data.count()} | Registros en Test: {test_data.count()}")

# Evaluadores
evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
evaluator_multi = MulticlassClassificationEvaluator(labelCol="isFraud", predictionCol="prediction")

# Nombres de las características predictoras (incluyendo variables generadas)
feature_names = ["Monto", "Saldo Inicial Origen", "Saldo Final Origen", "Saldo Inicial Destino", "Saldo Final Destino", "Error Saldo Origen", "Error Saldo Destino", "Tipo Transacción"]

# Función para calcular curva ROC manualmente de forma eficiente (sin scikit-learn)
def calculate_manual_roc(predictions_df):
    # Extraer la probabilidad de fraude (segundo elemento del vector de probabilidad)
    get_prob_udf = spark.udf.register("get_prob", lambda v: float(v[1]), DoubleType())
    prob_df = predictions_df.withColumn("prob_fraud", get_prob_udf("probability"))
    
    # Traer solo estas dos columnas a Pandas (muy ligero)
    pdf = prob_df.select("isFraud", "prob_fraud").toPandas()
    
    y_true = pdf["isFraud"].values
    y_prob = pdf["prob_fraud"].values
    
    # Ordenar por probabilidad descendente
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true = y_true[desc_score_indices]
    y_prob = y_prob[desc_score_indices]
    
    # Buscar cambios de umbral
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    # Sumas acumuladas de Verdaderos Positivos y Falsos Positivos
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # Agregar el punto inicial (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    
    return fpr, tpr

# =========================================================================
# ENTRENAMIENTO DE LOS 5 MODELOS
# =========================================================================

# --- MODELO 1: Logistic Regression ---
print("\n--- MODELO 1: Logistic Regression (LR) ---")
lr = LogisticRegression(labelCol="isFraud", featuresCol="features", maxIter=20, regParam=0.1)
print("Entrenando Logistic Regression...")
lr_model = lr.fit(train_data)

# --- MODELO 2: Naive Bayes ---
print("\n--- MODELO 2: Naive Bayes (NB) ---")
nb = NaiveBayes(labelCol="isFraud", featuresCol="features", smoothing=1.0)
print("Entrenando Naive Bayes...")
nb_model = nb.fit(train_data)

# --- MODELO 3: Decision Tree ---
print("\n--- MODELO 3: Decision Tree (DT) ---")
dt = DecisionTreeClassifier(labelCol="isFraud", featuresCol="features", maxDepth=8, seed=42)
print("Entrenando Decision Tree...")
dt_model = dt.fit(train_data)

# --- MODELO 4: Random Forest (RF) ---
print("\n--- MODELO 4: Random Forest (RF) ---")
rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", seed=42)
paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf.maxDepth, [5, 10])
                .addGrid(rf.numTrees, [10, 20])
                .build())
cv_rf = CrossValidator(estimator=rf,
                       estimatorParamMaps=paramGrid_rf,
                       evaluator=evaluator_auc,
                       numFolds=3,
                       seed=42)
print("Ajustando hiperparámetros y validando RandomForest...")
cv_model_rf = cv_rf.fit(train_data)
rf_model = cv_model_rf.bestModel
print(f"Mejor RF - maxDepth: {rf_model.getMaxDepth()}, numTrees: {rf_model.getNumTrees}")

# --- MODELO 5: Gradient Boosted Trees (GBTs) ---
print("\n--- MODELO 5: Gradient Boosted Trees (GBTs) ---")
gbt = GBTClassifier(labelCol="isFraud", featuresCol="features", seed=42)
paramGrid_gbt = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [5, 10])
                 .addGrid(gbt.maxIter, [10, 20])
                 .build())
cv_gbt = CrossValidator(estimator=gbt,
                        estimatorParamMaps=paramGrid_gbt,
                        evaluator=evaluator_auc,
                        numFolds=3,
                        seed=42)
print("Ajustando hiperparámetros y validando GBTs...")
cv_model_gbt = cv_gbt.fit(train_data)
gbt_model = cv_model_gbt.bestModel
print(f"Mejor GBT - maxDepth: {gbt_model.getMaxDepth()}, maxIter: {gbt_model.getMaxIter()}")

# =========================================================================
# EVALUACIÓN DE MODELOS Y EXTRACCIÓN DE MÉTRICAS
# =========================================================================
models = {
    "Logistic Regression": lr_model,
    "Naive Bayes": nb_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting (GBT)": gbt_model
}

results = {}
for name, model in models.items():
    print(f"\nEvaluando {name}...")
    predictions = model.transform(test_data)
    
    auc = evaluator_auc.evaluate(predictions)
    acc = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    prec = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
    rec = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
    f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
    
    print(f"{name} ROC-AUC:  {auc:.4f}")
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1-Score: {f1:.4f}")
    print(f"Matriz de Confusión {name}:")
    predictions.crosstab("isFraud", "prediction").show()
    
    results[name] = {
        "auc": auc,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "predictions": predictions
    }

# Guardar resultados en un archivo de texto
metrics_txt = "=== METRICAS DEL MODELO PREDICTIVO (SPARK) ===\nGenerado con PySpark en el cluster local/Docker.\n\n"
for name, res in results.items():
    metrics_txt += f"--- {name.upper()} ---\n"
    metrics_txt += f"Accuracy:  {res['acc']:.5f}\n"
    metrics_txt += f"Precision: {res['prec']:.5f}\n"
    metrics_txt += f"Recall:    {res['rec']:.5f}\n"
    metrics_txt += f"F1-Score:  {res['f1']:.5f}\n"
    metrics_txt += f"ROC-AUC:   {res['auc']:.5f}\n\n"

with open(os.path.join(OUT_DIR, "resultados_metricas_spark.txt"), "w") as f:
    f.write(metrics_txt)

# =========================================================================
# GENERACIÓN DE GRÁFICOS Y ANÁLISIS DE COMPARACIÓN
# =========================================================================

# 1. Gráfico de Comparación de Métricas
print("\nGenerando gráfico comparativo de métricas...")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_names))
width = 0.15
model_names = list(results.keys())

fig, ax = plt.subplots(figsize=(12, 7))
model_colors = ["#9467bd", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]

rects_list = []
for i, name in enumerate(model_names):
    res = results[name]
    values = [res["acc"], res["prec"], res["rec"], res["f1"], res["auc"]]
    pos = x + (i - 2) * width
    rects = ax.bar(pos, values, width, label=name, color=model_colors[i % len(model_colors)])
    rects_list.append(rects)

ax.set_ylabel('Puntuación (0.0 - 1.0)', fontsize=12)
ax.set_title('Comparación de Métricas: 5 Modelos de Machine Learning (Spark)', fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
ax.grid(axis='y', linestyle='--', alpha=0.5)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")

for rects in rects_list:
    autolabel(rects)

plt.tight_layout()
plt.savefig(PLOT_COMPARISON_PATH, dpi=150)
plt.close()
print(f"Gráfico comparativo de métricas guardado en: {PLOT_COMPARISON_PATH}")

# 2. Gráfico de Curvas ROC
print("Generando curvas ROC para los 5 modelos...")
plt.figure(figsize=(9, 7))

for i, name in enumerate(model_names):
    res = results[name]
    fpr, tpr = calculate_manual_roc(res["predictions"])
    plt.plot(fpr, tpr, color=model_colors[i % len(model_colors)], lw=2, label=f'{name} (AUC = {res["auc"]:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Clasificador Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
plt.title('Comparación de Curvas ROC - 5 Modelos (Spark)', fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(PLOT_ROC_PATH, dpi=150)
plt.close()
print(f"Gráfico de curvas ROC guardado en: {PLOT_ROC_PATH}")

# 3. Gráfico de Importancia de Variables (Feature Importance) del Mejor Modelo
best_model_name = max(results, key=lambda k: results[k]["f1"])
print(f"El mejor modelo según F1-Score es: {best_model_name}")

best_model = models[best_model_name]
if hasattr(best_model, "featureImportances"):
    print(f"Generando gráfico de importancia de variables para {best_model_name}...")
    importances = best_model.featureImportances.toArray()
else:
    alternative_model_name = "Gradient Boosting (GBT)" if "Gradient Boosting (GBT)" in results else "Random Forest"
    print(f"El mejor modelo ({best_model_name}) no tiene featureImportances. Usando {alternative_model_name}...")
    best_model = models[alternative_model_name]
    importances = best_model.featureImportances.toArray()
    best_model_name = alternative_model_name

indices = np.argsort(importances)

plt.figure(figsize=(9, 6))
plt.barh(range(len(indices)), importances[indices], color='#2ca02c', align='center', edgecolor='k', alpha=0.8)
plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=11)
plt.xlabel('Importancia Relativa', fontsize=12)
plt.title(f'Importancia de Variables en la Predicción de Fraude ({best_model_name})', fontsize=14, fontweight="bold", pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)

for index, value in enumerate(importances[indices]):
    plt.text(value + 0.005, index, f"{value:.4f}", va='center', ha='left', fontsize=10, fontweight="bold")

plt.xlim(0, max(importances) + 0.08)
plt.tight_layout()
plt.savefig(PLOT_IMPORTANCE_PATH, dpi=150)
plt.close()
print(f"Gráfico de importancia de variables guardado en: {PLOT_IMPORTANCE_PATH}")

# Liberar memoria
train_data.unpersist()
test_data.unpersist()

spark.stop()
print("\nProceso de entrenamiento y comparación completado de forma exitosa.")