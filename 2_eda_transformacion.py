from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col

# Iniciar Spark en Modo Local
spark = SparkSession.builder \
    .appName("Fraude_EDA_Transformacion") \
    .master("local[*]") \
    .getOrCreate()

# Leer data pre-procesada desde HDFS
print("Leyendo datos pre-procesados desde HDFS...")
df = spark.read.parquet("hdfs://namenode:9000/user/hadoop/fraud/fraude_preprocesado.parquet")

# 1. EDA: Estadísticas Descriptivas (Requisito Sección 8)
print("Análisis Exploratorio de montos y nuevas variables:")
df.describe(['amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrig', 'errorBalanceDest']).show()

# EDA: Conteo de la variable Objetivo
print("Desbalance de la variable objetivo (isFraud):")
df.groupBy("isFraud").count().show()

# 2. Transformación y Codificación
print("Codificando variable categórica 'type'...")
indexer = StringIndexer(inputCol="type", outputCol="typeIndex")
df_indexed = indexer.fit(df).transform(df)

encoder = OneHotEncoder(inputCols=["typeIndex"], outputCols=["typeVec"])
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# 3. Ensamblado de Vectores (Incluye las variables errorBalance)
print("Ensamblando vectores de características...")
assembler = VectorAssembler(
    inputCols=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "typeVec"],
    outputCol="features"
)
df_final = assembler.transform(df_encoded)

# Guardar Dataset listo para ML en Parquet en HDFS
print("Guardando dataset listo para ML en Parquet en HDFS...")
df_final.select("features", "isFraud").write.mode("overwrite").parquet("hdfs://namenode:9000/user/hadoop/fraud/fraude_ml_ready.parquet")

print("Paso 2 completado con éxito. Cerrando sesión de Spark.")
spark.stop()