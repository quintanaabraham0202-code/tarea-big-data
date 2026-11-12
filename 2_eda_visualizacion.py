import os
import matplotlib
matplotlib.use("Agg")  # No GUI needed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# 1. Configuración de directorios y Spark
OUT_DIR = "/app/graficas_fraude"
os.makedirs(OUT_DIR, exist_ok=True)

print("Iniciando SparkSession para Visualizaciones EDA...")
spark = SparkSession.builder \
    .appName("Fraude_EDA_Visualizacion") \
    .getOrCreate()

# Cargar dataset desde HDFS
print("Cargando dataset desde Hadoop HDFS...")
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("hdfs://namenode:9000/user/hadoop/fraud/fraud_dataset.csv")

# Paleta de colores Premium
COLOR_NORMAL = "#1f77b4"      # Azul
COLOR_FRAUD = "#d62728"       # Rojo
COLOR_ACCENT = "#9467bd"      # Púrpura
COLOR_SECONDARY = "#2ca02c"   # Verde
COLORS_PALETTE = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"]

# ---------------------------------------------------------
# GRAFICO 1: Distribución de transacciones por tipo (type)
# ---------------------------------------------------------
print("Graficando 1: Distribución de transacciones por tipo...")
df_type = df.groupBy("type").count().orderBy("count", ascending=False).toPandas()
plt.figure(figsize=(8, 5))
bars = plt.bar(df_type["type"], df_type["count"], color=COLORS_PALETTE[:len(df_type)])
plt.title("Distribución de Transacciones por Tipo", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Tipo de Transacción", fontsize=12)
plt.ylabel("Cantidad (Escala Lineal)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f"{height:,}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_01_tipo_transacciones.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 2: Cantidad de fraudes vs. no fraudes (isFraud)
# ---------------------------------------------------------
print("Graficando 2: Cantidad de fraudes vs. no fraudes...")
df_fraud = df.groupBy("isFraud").count().toPandas()
df_fraud["Estado"] = df_fraud["isFraud"].map({0: "No Fraude", 1: "Fraude"})
plt.figure(figsize=(6, 5))
bars = plt.bar(df_fraud["Estado"], df_fraud["count"], color=[COLOR_NORMAL, COLOR_FRAUD])
plt.title("Desbalance de la Variable Objetivo (isFraud)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Clase", fontsize=12)
plt.ylabel("Cantidad (Escala Logarítmica)", fontsize=12)
plt.yscale("log")  # Escala logarítmica para ver la clase minoritaria
plt.grid(axis='y', linestyle='--', alpha=0.5, which="both")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height * 1.1, f"{height:,}", ha='center', va='bottom', fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_02_balance_fraude.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 3: Fraudes por tipo de transacción
# ---------------------------------------------------------
print("Graficando 3: Fraudes por tipo de transacción...")
df_type_fraud = df.groupBy("type", "isFraud").count().orderBy("type").toPandas()
pivot_df = df_type_fraud.pivot(index='type', columns='isFraud', values='count').fillna(0)
pivot_df.columns = ["No Fraude", "Fraude"]
plt.figure(figsize=(9, 5))
pivot_df.plot(kind="bar", color=[COLOR_NORMAL, COLOR_FRAUD], logy=True, figsize=(10, 6))
plt.title("Transacciones Normales vs Fraudulentas por Tipo", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Tipo de Transacción", fontsize=12)
plt.ylabel("Cantidad (Escala Logarítmica)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5, which="both")
plt.legend(title="Estado")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_03_fraude_por_tipo.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# PREPARACIÓN DE MUESTRAS PARA GRÁFICOS CONTINUOS
# ---------------------------------------------------------
print("Preparando muestras híbridas de datos...")
df_normal_all = df.filter(col("isFraud") == 0)
df_fraud_all = df.filter(col("isFraud") == 1)
df_normal_sample = df_normal_all.sample(False, 0.02, seed=42)
df_combined = df_normal_sample.union(df_fraud_all).toPandas()
print(f"Muestra híbrida creada: {len(df_combined)} registros.")

# ---------------------------------------------------------
# GRAFICO 4: Distribución del monto (amount) (Histograma)
# ---------------------------------------------------------
print("Graficando 4: Distribución de montos...")
plt.figure(figsize=(9, 5))
plt.hist(df_combined["amount"], bins=50, color=COLOR_NORMAL, edgecolor='k', alpha=0.7)
plt.title("Distribución General del Monto de Transacciones", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Monto (amount)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_04_distribucion_monto.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 5: Boxplot del monto según fraude
# ---------------------------------------------------------
print("Graficando 5: Boxplot del monto por fraude...")
plt.figure(figsize=(7, 6))
data_to_plot = [df_combined[df_combined["isFraud"] == 0]["amount"], 
                df_combined[df_combined["isFraud"] == 1]["amount"]]
plt.boxplot(data_to_plot, patch_artist=True,
            boxprops=dict(facecolor="#aec7e8", color=COLOR_NORMAL),
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='r', markersize=3, alpha=0.3))
plt.xticks([1, 2], ["No Fraude", "Fraude"])
plt.title("Comparación de Montos según Estado de Fraude", fontsize=14, fontweight="bold", pad=15)
plt.ylabel("Monto (Escala Logarítmica)", fontsize=12)
plt.yscale("log")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_05_boxplot_monto_fraude.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 6 y 7: Evolución temporal por step (Línea)
# ---------------------------------------------------------
print("Graficando 6 y 7: Evolución temporal...")
step_counts_total = df.groupBy("step").count().orderBy("step").toPandas()
step_counts_fraud = df.filter(col("isFraud") == 1).groupBy("step").count().orderBy("step").toPandas()

plt.figure(figsize=(10, 5))
plt.plot(step_counts_total["step"], step_counts_total["count"], color=COLOR_NORMAL, linewidth=1.5)
plt.title("Evolución Temporal del Volumen Total de Transacciones", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Tiempo (step en horas)", fontsize=12)
plt.ylabel("Cantidad de Transacciones", fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_06_evolucion_tiempo_total.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(step_counts_fraud["step"], step_counts_fraud["count"], color=COLOR_FRAUD, linewidth=1.5, marker='o', markersize=2)
plt.title("Evolución Temporal de Casos de Fraude", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Tiempo (step en horas)", fontsize=12)
plt.ylabel("Cantidad de Fraudes", fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_07_evolucion_tiempo_fraude.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 8: Correlación entre variables numéricas (Mapa de calor)
# ---------------------------------------------------------
print("Graficando 8: Mapa de calor de correlación...")
num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
corr_matrix = df_combined[num_cols].corr()

plt.figure(figsize=(8, 7))
im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(np.arange(len(num_cols)), num_cols, rotation=45, ha="right")
plt.yticks(np.arange(len(num_cols)), num_cols)
plt.title("Matriz de Correlación (Pearson)", fontsize=14, fontweight="bold", pad=15)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        text = plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.6 else "white",
                       fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_08_mapa_calor_correlacion.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 9: Dispersión: Monto vs Saldo inicial
# ---------------------------------------------------------
print("Graficando 9: Dispersión Monto vs Saldo inicial...")
plt.figure(figsize=(9, 6))
normal_data = df_combined[df_combined["isFraud"] == 0]
fraud_data = df_combined[df_combined["isFraud"] == 1]

plt.scatter(normal_data["oldbalanceOrg"], normal_data["amount"], c=COLOR_NORMAL, alpha=0.4, label="No Fraude", s=10)
plt.scatter(fraud_data["oldbalanceOrg"], fraud_data["amount"], c=COLOR_FRAUD, alpha=0.8, label="Fraude", s=15, edgecolors='k', linewidths=0.3)

plt.title("Patrones de Fraude: Monto vs Saldo Origen Inicial", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Saldo Inicial de Origen (oldbalanceOrg)", fontsize=12)
plt.ylabel("Monto de la Transacción (amount)", fontsize=12)
plt.xscale("symlog")
plt.yscale("symlog")
plt.grid(linestyle='--', alpha=0.5)
plt.legend(title="Clase")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_09_dispersion_monto_saldo.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 10: Distribución del saldo del origen (oldbalanceOrg)
# ---------------------------------------------------------
print("Graficando 10: Distribución del saldo inicial de origen...")
plt.figure(figsize=(9, 5))
plt.hist(df_combined["oldbalanceOrg"], bins=50, color=COLOR_ACCENT, edgecolor='k', alpha=0.7, log=True)
plt.title("Distribución del Saldo Origen Inicial (oldbalanceOrg)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Saldo Inicial", fontsize=12)
plt.ylabel("Frecuencia (Escala Logarítmica)", fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_10_distribucion_saldo_origen.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 11: Distribución del saldo del destino
# ---------------------------------------------------------
print("Graficando 11: Distribución comparativa del saldo de destino...")
plt.figure(figsize=(9, 5))
plt.hist(df_combined["oldbalanceDest"], bins=50, color="#aec7e8", alpha=0.6, label="Saldo Inicial Destino", log=True)
plt.hist(df_combined["newbalanceDest"], bins=50, color="#ffbb78", alpha=0.6, label="Saldo Final Destino", log=True)
plt.title("Comparación de Saldos en Cuenta Destino", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Saldo Destino", fontsize=12)
plt.ylabel("Frecuencia (Escala Logarítmica)", fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_11_distribucion_saldo_destino.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICO 12: Porcentaje de transacciones marcadas como sospechosas (isFlaggedFraud)
# ---------------------------------------------------------
print("Graficando 12: Transacciones marcadas como sospechosas...")
df_flagged = df.groupBy("isFlaggedFraud").count().toPandas()
df_flagged["Sospechosa"] = df_flagged["isFlaggedFraud"].map({0: "No Sospechosa", 1: "Sospechosa (Flagged)"})

plt.figure(figsize=(6, 5))
bars = plt.bar(df_flagged["Sospechosa"], df_flagged["count"], color=[COLOR_NORMAL, COLOR_ACCENT], log=True)
plt.title("Transacciones Marcadas por el Sistema (isFlaggedFraud)", fontsize=14, fontweight="bold", pad=15)
plt.ylabel("Frecuencia (Escala Logarítmica)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5, which="both")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height * 1.1, f"{height:,}", ha='center', va='bottom', fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_12_sospechosas.png"), dpi=150)
plt.close()

print("¡Todas las visualizaciones EDA locales generadas y guardadas en graficas_fraude!")
spark.stop()
