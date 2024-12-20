# Research Gaps Identification with Graphs

A scalable Python application for identifying research gaps using keyword co-occurrence analysis with Apache Spark and GPU acceleration.

## Features

- Distributed processing with Apache Spark
- GPU acceleration
- N-gram analysis (unigrams, bigrams, trigrams)
- Graph-based research gap identification
- Interactive visualizations
- Configurable parameters and thresholds
- Support for both CSV and JSON input formats

## Requirements

- Python 3.8+
- CUDA-capable GPU
- Apache Spark 3.2+
- 8GB+ RAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SandeepChatterjee66/research-gaps.git
cd research-gaps
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
research_gaps/
├── config/
│   └── config.py           # Configuration classes
├── data/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   └── data_cleaner.py     # Data cleaning utilities
├── preprocessing/
│   ├── __init__.py
│   └── text_processor.py   # Text processing utilities
├── core/
│   ├── __init__.py
│   └── processors.py       # Core processing modules
├── utils/
│   ├── __init__.py
│   └── utils.py           # Utility functions
├── visualization/
│   ├── __init__.py
│   └── visualizer.py      # Visualization utilities
└── main.py                # Main execution script
```

PS : this was old structure, all the files are in root now.

## Configuration

Edit `config/config.py` to modify:
- Spark configuration (memory, cores)
- Project parameters (thresholds, limits)
- Stopwords and filtering criteria

Example configuration:
```python
@dataclass
class ProjectConfig:
    n_keywords: int = 5000
    alpha: float = 0.5
    k: int = 10
    min_keyword_freq: int = 5
    similarity_threshold: float = 0.9
```

## Usage

1. Prepare your input data in CSV format with columns:
   - id: unique identifier
   - keywords: array of keywords

Note : Dataset upload has failed due to error in commits but few data are there

2. Update the data path in `main.py`:
```python
df = loader.load_csv("datasets/data.csv", "keywords")
```

3. Run the analysis:
```bash
python main.py
```

The script will:
- Load and clean the data
- Process n-grams
- Build co-occurrence graph
- Identify research gaps
- Generate visualizations

## Output

The program generates:
1. Keyword co-occurrence graph (`keyword_graph.png`)
2. Research gap triangles (returned by main function)

## Example

```python
from config.config import SparkConfig, ProjectConfig
from main import main

# Run analysis
gaps = main()

# Print top research gaps
for (kw1, kw2, kw3), score in gaps:
    print(f"Research Gap: {kw1} - {kw2} - {kw3} (Score: {score:.2f})")
```

## GPU Acceleration

The application uses CuPy for GPU-accelerated computations. Ensure you have:
1. CUDA toolkit installed
2. Compatible GPU drivers
3. CuPy installed with appropriate CUDA version

## Performance Tips

1. Adjust Spark configuration based on your hardware:
```python
@dataclass
class SparkConfig:
    executor_memory: str = "4g"  # Increase for larger datasets
    driver_memory: str = "4g"    # Increase for larger datasets
```

2. Modify batch sizes for GPU processing in `text_processor.py`
3. Use appropriate `min_keyword_freq` threshold to filter noise

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Sandeep Chatterjee
Project Link: https://github.com/SandeepChatterjee66/Research-Gap-Identification
