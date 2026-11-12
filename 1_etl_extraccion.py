import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan

# 1. Iniciar Spark en Modo Local
print("Iniciando SparkSession Local para Detección de Fraude...")
spark = SparkSession.builder \
    .appName("Fraude_ETL_Paso1") \
    .master("local[*]") \
    .getOrCreate()

# 2. Cargar CSV desde Hadoop HDFS
print("Cargando datos desde Hadoop HDFS...")
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("hdfs://namenode:9000/user/hadoop/fraud/fraud_dataset.csv")

# 3. Limpieza de Valores Nulos e Inválidos (Tratamiento de nulos y Outliers)
print("Buscando valores nulos iniciales...")
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

print("Limpiando valores nulos y eliminando transacciones con montos inválidos (amount <= 0)...")
df_clean = df.dropna(how="any", subset=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]) \
             .filter(col("amount") > 0)

# 4. Filtrar datos (Regla de negocio: TRANSFER y CASH_OUT)
print("Filtrando transacciones relevantes (TRANSFER y CASH_OUT)...")
df_valid = df_clean.filter((col("type") == "TRANSFER") | (col("type") == "CASH_OUT"))

# 5. Generación de Nuevas Variables (Feature Engineering - Sección 7)
print("Generando nuevas variables predictivas (errorBalanceOrig y errorBalanceDest)...")
df_features = df_valid.withColumn(
    "errorBalanceOrig", 
    col("oldbalanceOrg") - col("amount") - col("newbalanceOrig")
).withColumn(
    "errorBalanceDest", 
    col("newbalanceDest") - col("oldbalanceDest") - col("amount")
)

# 6. Guardado Intermedio en formato Parquet en HDFS con Particionado Físico
print("Guardando datos pre-procesados en HDFS (Formato Parquet particionado)...")
df_features.write.partitionBy("type").mode("overwrite").parquet("hdfs://namenode:9000/user/hadoop/fraud/fraude_preprocesado.parquet")

print("Paso 1 completado con éxito. Cerrando sesión de Spark.")
spark.stop()