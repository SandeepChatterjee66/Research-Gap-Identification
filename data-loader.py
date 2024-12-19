from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import pandas as pd

class DataLoader:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def load_csv(self, path: str, keyword_column: str):
        """Load dataset from CSV file"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField(keyword_column, ArrayType(StringType()), True)
        ])
        
        return (self.spark.read
                .option("header", "true")
                .schema(schema)
                .csv(path))
    
    def load_json(self, path: str):
        """Load dataset from JSON file"""
        return self.spark.read.json(path)
