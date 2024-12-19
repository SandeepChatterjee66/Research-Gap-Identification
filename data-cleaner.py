from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, size
from pyspark.sql.types import ArrayType, StringType
import re
from typing import List
from config.config import ProjectConfig

class DataCleaner:
    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark = spark
        self.config = config
        
    @staticmethod
    @udf(returnType=ArrayType(StringType()))
    def clean_keywords(keywords: List[str]) -> List[str]:
        """Clean and normalize keywords"""
        if not keywords:
            return []
            
        cleaned = []
        for kw in keywords:
            # Convert to lowercase
            kw = kw.lower().strip()
            
            # Remove special characters
            kw = re.sub(r'[^\w\s-]', '', kw)
            
            # Remove extra whitespace
            kw = re.sub(r'\s+', ' ', kw).strip()
            
            if kw:
                cleaned.append(kw)
                
        return cleaned
    
    def clean_dataset(self, df, keyword_column: str):
        """Clean the entire dataset"""
        return (df.withColumn(keyword_column, 
                            self.clean_keywords(col(keyword_column)))
                .filter(col(keyword_column).isNotNull())
                .filter(size(col(keyword_column)) > 0))
