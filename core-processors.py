from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, count, col, arrays_zip
from pyspark.sql.types import ArrayType, StructType, StructField, StringType
import networkx as nx
import cupy as cp
from typing import List, Dict, Tuple
from config.config import ProjectConfig

class UnigramProcessor:
    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark = spark
        self.config = config
        
    def process_unigrams(self, df, keyword_column: str):
        """Process unigrams using Spark"""
        return (df.select(explode(col(keyword_column)).alias("keyword"))
                .groupBy("keyword")
                .agg(count("*").alias("frequency"))
                .filter(col("frequency") >= self.config.min_keyword_freq)
                .orderBy(col("frequency").desc())
                .limit(self.config.n_keywords))

class BigramProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def process_bigrams(self, df, keyword_column: str):
        """Process bigrams using Spark and GPU acceleration"""
        bigram_df = df.selectExpr(
            f"inline(flatten(transform({keyword_column}, x -> " +
            f"transform({keyword_column}, y -> struct(x as word1, y as word2))))")
        
        return (bigram_df
                .filter(col("word1") < col("word2"))
                .groupBy("word1", "word2")
                .count()
                .orderBy(col("count").desc()))

class TrigramProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def process_trigrams(self, df, keyword_column: str):
        """Process trigrams using Spark and GPU acceleration"""
        trigram_expr = (
            f"inline(flatten(transform({keyword_column}, x -> " +
            f"transform({keyword_column}, y -> " +
            f"transform({keyword_column}, z -> " +
            f"struct(x as word1, y as word2, z as word3)))))"
        )
        
        return (df.selectExpr(trigram_expr)
                .filter((col("word1") < col("word2")) & 
                        (col("word2") < col("word3")))
                .groupBy("word1", "word2", "word3")
                .count()
                .orderBy(col("count").desc()))

class GraphProcessor:
    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark = spark
        self.config = config
        
    def build_graph(self, bigram_counts):
        """Build co-occurrence graph using NetworkX"""
        G = nx.Graph()
        edges = bigram_counts.select("word1", "word2", "count").collect()
        
        for row in edges:
            G.add_edge(row.word1, row.word2, weight=row.count)
            
        return G
    
    def find_triangles(self, G: nx.Graph, trigram_counts):
        """Find and rank triangles using GPU acceleration"""
        trigram_dict = {(row.word1, row.word2, row.word3): row.count 
                       for row in trigram_counts.collect()}
        
        triangles = []
        for t1, t2, t3 in nx.triangles(G):
            W_triangle = (
                G[t1][t2]["weight"] +
                G[t2][t3]["weight"] +
                G[t1][t3]["weight"]
            )
            
            triangle = tuple(sorted([t1, t2, t3]))
            F_triangle = trigram_dict.get(triangle, 1)
            S_triangle = W_triangle / (F_triangle ** self.config.alpha)
            
            triangles.append((triangle, S_triangle))
            
        return sorted(triangles, key=lambda x: x[1], reverse=True)[:self.config.k]
