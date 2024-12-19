from dataclasses import dataclass
from typing import List, Set

@dataclass
class SparkConfig:
    app_name: str = "ResearchGapsIdentifier"
    master: str = "local[*]"
    executor_memory: str = "4g"
    driver_memory: str = "4g"
    
@dataclass
class ProjectConfig:
    n_keywords: int = 5000
    alpha: float = 0.5
    k: int = 10
    min_keyword_freq: int = 5
    similarity_threshold: float = 0.9
    stopwords: Set[str] = frozenset({
        'method', 'analysis', 'study', 'approach', 'using',
        'based', 'paper', 'research', 'new', 'proposed'
    })
