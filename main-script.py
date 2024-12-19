from config.config import SparkConfig, ProjectConfig
from utils.utils import create_spark_session
from data.data_loader import DataLoader
from data.data_cleaner import DataCleaner
from preprocessing.text_processor import TextProcessor
from core.processors import (
    UnigramProcessor, 
    BigramProcessor, 
    TrigramProcessor, 
    GraphProcessor
)
from visualization.visualizer import Visualizer

def main():
    # Initialize configurations
    spark_config = SparkConfig()
    project_config = ProjectConfig()
    
    # Create Spark session
    spark = create_spark_session(spark_config)
    
    # Initialize components
    loader = DataLoader(spark)
    cleaner = DataCleaner(spark, project_config)
    text_processor = TextProcessor(spark)
    unigram_processor = UnigramProcessor(spark, project_config)
    bigram_processor = BigramProcessor(spark)
    trigram_processor = TrigramProcessor(spark)
    graph_processor = GraphProcessor(spark, project_config)
    
    try:
        # Load and process data
        df = loader.load_csv("path/to/your/data.csv", "keywords")
        cleaned_df = cleaner.clean_dataset(df, "keywords")
        
        # Process n-grams
        unigram_counts = unigram_processor.process_unigrams(cleaned_df, "keywords")
        bigram_counts = bigram_processor.process_bigrams(cleaned_df, "keywords")
        trigram_counts = trigram_processor.process_trigrams(cleaned_df, "keywords")
        
        # Build graph and find research gaps
        G = graph_processor.build_graph(bigram_counts)
        gaps = graph_processor.find_triangles(G, trigram_counts)
        
        # Visualize results
        visualizer = Visualizer()
        graph_plot = visualizer.plot_keyword_graph(G)
        graph_plot.savefig("keyword_graph.png")
        
        print("Analysis completed successfully!")
        return gaps
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
