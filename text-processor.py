from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import numpy as np
import cupy as cp
from typing import List

class TextProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    @staticmethod
    @udf(returnType=FloatType())
    def levenshtein_distance_gpu(s1: str, s2: str) -> float:
        """GPU-accelerated Levenshtein distance computation"""
        if not s1 or not s2:
            return 1.0
            
        # Convert strings to character arrays
        s1_arr = cp.array([ord(c) for c in s1])
        s2_arr = cp.array([ord(c) for c in s2])
        
        # Create distance matrix
        rows = len(s1) + 1
        cols = len(s2) + 1
        dist_matrix = cp.zeros((rows, cols), dtype=cp.int32)
        
        # Initialize first row and column
        dist_matrix[0, :] = cp.arange(cols)
        dist_matrix[:, 0] = cp.arange(rows)
        
        # Compute distances
        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if s1_arr[i-1] == s2_arr[j-1] else 1
                dist_matrix[i, j] = min(
                    dist_matrix[i-1, j] + 1,      # deletion
                    dist_matrix[i, j-1] + 1,      # insertion
                    dist_matrix[i-1, j-1] + cost  # substitution
                )
        
        # Normalize
        return float(1.0 - (2.0 * dist_matrix[-1, -1] / (len(s1) + len(s2))))
