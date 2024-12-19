from pyspark.sql import SparkSession
from config.config import SparkConfig
import cupy as cp
import numpy as np
from typing import Any

def create_spark_session(config: SparkConfig) -> SparkSession:
    """Create and configure Spark session"""
    return (SparkSession.builder
            .appName(config.app_name)
            .master(config.master)
            .config("spark.executor.memory", config.executor_memory)
            .config("spark.driver.memory", config.driver_memory)
            .config("spark.rapids.sql.enabled", "true")  # Enable RAPIDS acceleration
            .getOrCreate())

def to_gpu(data: Any) -> Any:
    """Transfer data to GPU"""
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    elif isinstance(data, list):
        return cp.asarray(np.array(data))
    return data

def from_gpu(data: Any) -> Any:
    """Transfer data from GPU"""
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data
