# Visualization Module

## Overview

The visualization module (`rl_ids.plots`) provides comprehensive plotting and analysis tools for the RL-IDS system. It generates publication-quality visualizations for training metrics, evaluation results, class distributions, error analysis, and model performance insights.

## Key Features

- **Training Metrics Visualization** - Comprehensive training progress plots
- **Evaluation Analytics** - Model performance and confusion matrix analysis
- **Class Distribution Analysis** - Detailed class imbalance and prediction patterns
- **Error Analysis** - Misclassification patterns and confidence analysis
- **Publication-Quality Output** - High-resolution, customizable plots
- **Batch Plot Generation** - Automated report generation from CSV files
- **Interactive CLI** - Command-line interface for plot generation

## Classes

### `IDSPlotter`

Main plotting class providing comprehensive visualization capabilities.

**Constructor**

```python
IDSPlotter(figures_dir: Path = FIGURES_DIR, dpi: int = 300)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `figures_dir` | `Path` | Directory to save generated figures |
| `dpi` | `int` | Resolution for saved figures (default: 300) |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `figures_dir` | `Path` | Output directory for plots |
| `dpi` | `int` | Image resolution |
| `colors` | `dict` | Color scheme for consistent styling |
| `class_colors` | `list` | Color palette for class-based plots |

## Core Methods

### Training Visualization

#### `plot_training_metrics(metrics_path: Union[str, Path], save_name: str = "training_metrics_overview") -> None`

Creates comprehensive training metrics visualization including accuracy, reward, loss, and epsilon decay.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `metrics_path` | `Union[str, Path]` | Path to training metrics CSV file |
| `save_name` | `str` | Base name for saved plot files |

**Generated Plots**
- Training accuracy progression with mean/max lines
- Training reward with moving average
- Loss curves (Q-loss, policy loss if available)
- Epsilon decay schedule
- Learning rate progression
- Episode duration analysis

**Example**
```python
from rl_ids.plots import IDSPlotter
from rl_ids.config import REPORTS_DIR, FIGURES_DIR

plotter = IDSPlotter(FIGURES_DIR, dpi=300)

# Plot training metrics
plotter.plot_training_metrics(
    metrics_path=REPORTS_DIR / "training_metrics.csv",
    save_name="dqn_training_overview"
)
```

#### `plot_episode_analysis(metrics_path: Union[str, Path], save_name: str = "episode_analysis") -> None`

Analyzes episode-level performance patterns and convergence behavior.

**Generated Plots**
- Episode reward distribution
- Episode length analysis
- Convergence detection
- Performance stability metrics

### Evaluation Visualization

#### `plot_evaluation_overview(reports_dir: Path, save_name: str = "evaluation_overview") -> None`

Creates comprehensive evaluation analysis from generated CSV reports.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `reports_dir` | `Path` | Directory containing evaluation CSV files |
| `save_name` | `str` | Base name for saved plot files |

**Required CSV Files**
- `evaluation_summary.csv` - Overall performance metrics
- `evaluation_confusion_matrix.csv` - Confusion matrix data
- `evaluation_detailed_predictions.csv` - Per-sample predictions
- `evaluation_classification_report.csv` - Per-class metrics

**Generated Plots**
- Overall accuracy and performance metrics
- Confusion matrix heatmap
- Per-class precision, recall, F1-score
- Prediction confidence distribution
- ROC curves (if binary classification)

**Example**
```python
# Generate evaluation overview from reports
plotter.plot_evaluation_overview(
    reports_dir=REPORTS_DIR,
    save_name="model_evaluation_complete"
)
```

#### `plot_confusion_matrix(cm_data: Union[np.ndarray, pd.DataFrame], class_names: list = None, save_name: str = "confusion_matrix") -> None`

Creates enhanced confusion matrix visualization.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `cm_data` | `Union[np.ndarray, pd.DataFrame]` | Confusion matrix data |
| `class_names` | `list` | List of class names for labels |
| `save_name` | `str` | Name for saved plot file |

**Features**
- Normalized and raw count versions
- Per-class accuracy annotations
- Color-coded intensity mapping
- Statistical summary overlay

### Class Analysis

#### `plot_class_analysis(reports_dir: Path, save_name: str = "class_analysis") -> None`

Analyzes class distribution and prediction patterns.

**Generated Plots**
- Class distribution in training/test sets
- Per-class prediction accuracy
- Class imbalance impact analysis
- Misclassification patterns between classes

#### `plot_class_distribution(data_path: Union[str, Path], save_name: str = "class_distribution") -> None`

Visualizes class distribution in the dataset.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data_path` | `Union[str, Path]` | Path to dataset CSV file |
| `save_name` | `str` | Name for saved plot file |

**Generated Plots**
- Bar chart of class frequencies
- Pie chart of class proportions
- Class imbalance ratio analysis
- Logarithmic scale view for extreme imbalance

### Error Analysis

#### `plot_error_analysis(reports_dir: Path, save_name: str = "error_analysis") -> None`

Performs detailed error analysis and misclassification patterns.

**Generated Plots**
- Error distribution by class
- Confidence vs. accuracy correlation
- Feature importance for errors
- Temporal error patterns

#### `plot_confidence_analysis(predictions_path: Union[str, Path], save_name: str = "confidence_analysis") -> None`

Analyzes model prediction confidence patterns.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `predictions_path` | `Union[str, Path]` | Path to detailed predictions CSV |
| `save_name` | `str` | Name for saved plot file |

**Generated Plots**
- Confidence distribution by correctness
- Confidence vs. accuracy relationship
- Low-confidence prediction analysis
- Calibration curves

### Comparison and Benchmarking

#### `plot_model_comparison(results_dict: dict, save_name: str = "model_comparison") -> None`

Compares multiple model results side-by-side.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `results_dict` | `dict` | Dictionary mapping model names to result paths |
| `save_name` | `str` | Name for saved plot file |

**Example**
```python
model_results = {
    'DQN_v1': 'reports/dqn_v1/',
    'DQN_v2': 'reports/dqn_v2/', 
    'DQN_optimized': 'reports/dqn_final/'
}

plotter.plot_model_comparison(
    results_dict=model_results,
    save_name="model_versions_comparison"
)
```

## Utility Methods

### Data Loading and Processing

#### `load_evaluation_data(reports_dir: Path) -> dict`

Loads and validates evaluation data from CSV files.

**Returns**
- `dict`: Dictionary containing loaded DataFrames for each report type

#### `validate_data_format(df: pd.DataFrame, expected_columns: list) -> bool`

Validates CSV data format for plotting compatibility.

### Plot Styling and Customization

#### `apply_plot_style(ax: plt.Axes, title: str = None) -> None`

Applies consistent styling to matplotlib axes.

#### `save_plot(fig: plt.Figure, filename: str, tight_layout: bool = True) -> Path`

Saves plots with consistent formatting and metadata.

## Command Line Interface

Generate plots directly from the command line:

```bash
# Generate all evaluation plots
python -m rl_ids.plots evaluation-overview --reports-dir ./reports

# Generate training metrics plots
python -m rl_ids.plots training-metrics --metrics-file ./reports/training_metrics.csv

# Generate class analysis
python -m rl_ids.plots class-analysis --reports-dir ./reports

# Generate error analysis
python -m rl_ids.plots error-analysis --reports-dir ./reports

# Custom output directory and DPI
python -m rl_ids.plots evaluation-overview \
    --reports-dir ./reports \
    --output-dir ./custom_figures \
    --dpi 600

# Generate specific plot types
python -m rl_ids.plots confusion-matrix \
    --matrix-file ./reports/evaluation_confusion_matrix.csv \
    --class-names-file ./reports/class_names.txt
```

### CLI Commands

| Command | Description | Options |
|---------|-------------|---------|
| `evaluation-overview` | Generate complete evaluation analysis | `--reports-dir`, `--output-dir`, `--dpi` |
| `training-metrics` | Plot training progress | `--metrics-file`, `--output-dir`, `--dpi` |
| `class-analysis` | Analyze class distributions | `--reports-dir`, `--data-file`, `--output-dir` |
| `error-analysis` | Error pattern analysis | `--reports-dir`, `--predictions-file`, `--output-dir` |
| `confusion-matrix` | Enhanced confusion matrix | `--matrix-file`, `--class-names-file`, `--output-dir` |
| `confidence-analysis` | Prediction confidence analysis | `--predictions-file`, `--output-dir`, `--dpi` |

## Complete Visualization Workflow

### Example: Full Report Generation

```python
from rl_ids.plots import IDSPlotter
from rl_ids.config import REPORTS_DIR, FIGURES_DIR
from pathlib import Path

def generate_complete_report():
    """Generate complete visualization report"""
    
    # Initialize plotter
    plotter = IDSPlotter(FIGURES_DIR, dpi=300)
    
    # 1. Training Analysis
    if (REPORTS_DIR / "training_metrics.csv").exists():
        plotter.plot_training_metrics(
            REPORTS_DIR / "training_metrics.csv",
            save_name="training_complete"
        )
        
        plotter.plot_episode_analysis(
            REPORTS_DIR / "training_metrics.csv",
            save_name="episode_analysis"
        )
    
    # 2. Evaluation Analysis
    plotter.plot_evaluation_overview(
        REPORTS_DIR,
        save_name="evaluation_complete"
    )
    
    # 3. Class Analysis
    plotter.plot_class_analysis(
        REPORTS_DIR,
        save_name="class_analysis_detailed"
    )
    
    # 4. Error Analysis
    plotter.plot_error_analysis(
        REPORTS_DIR,
        save_name="error_analysis_detailed"
    )
    
    # 5. Confidence Analysis
    if (REPORTS_DIR / "evaluation_detailed_predictions.csv").exists():
        plotter.plot_confidence_analysis(
            REPORTS_DIR / "evaluation_detailed_predictions.csv",
            save_name="confidence_analysis"
        )
    
    # 6. Generate summary report
    generate_summary_report(FIGURES_DIR)
    
    print(f"Complete report generated in: {FIGURES_DIR}")

def generate_summary_report(figures_dir: Path):
    """Generate HTML summary report with all plots"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RL-IDS Model Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .plot-section { margin: 30px 0; }
            .plot-image { max-width: 100%; height: auto; }
            h1, h2 { color: #2E86AB; }
        </style>
    </head>
    <body>
        <h1>RL-IDS Model Analysis Report</h1>
        
        <h2>Training Analysis</h2>
        <div class="plot-section">
            <img src="training_complete.png" class="plot-image" alt="Training Metrics">
        </div>
        
        <h2>Evaluation Results</h2>
        <div class="plot-section">
            <img src="evaluation_complete.png" class="plot-image" alt="Evaluation Overview">
        </div>
        
        <h2>Class Analysis</h2>
        <div class="plot-section">
            <img src="class_analysis_detailed.png" class="plot-image" alt="Class Analysis">
        </div>
        
        <h2>Error Analysis</h2>
        <div class="plot-section">
            <img src="error_analysis_detailed.png" class="plot-image" alt="Error Analysis">
        </div>
        
        <h2>Confidence Analysis</h2>
        <div class="plot-section">
            <img src="confidence_analysis.png" class="plot-image" alt="Confidence Analysis">
        </div>
    </body>
    </html>
    """
    
    with open(figures_dir / "analysis_report.html", 'w') as f:
        f.write(html_content)

# Generate complete report
generate_complete_report()
```

## Custom Plot Themes

### Theme Configuration

```python
# Custom color schemes
custom_colors = {
    'corporate': {
        'primary': '#003f5c',
        'secondary': '#58508d', 
        'accent': '#bc5090',
        'warning': '#ff6361',
        'success': '#ffa600'
    },
    'academic': {
        'primary': '#1f4e79',
        'secondary': '#2e8b57',
        'accent': '#d2691e',
        'warning': '#dc143c',
        'success': '#228b22'
    }
}

# Apply custom theme
plotter = IDSPlotter(FIGURES_DIR)
plotter.colors = custom_colors['academic']
```

### Advanced Styling

```python
def setup_publication_style():
    """Setup matplotlib for publication-quality plots"""
    
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,  # Set to True if LaTeX is available
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': False
    })

# Apply before plotting
setup_publication_style()
plotter = IDSPlotter(FIGURES_DIR, dpi=600)
```

## Performance Optimization

### Large Dataset Visualization

```python
def plot_large_dataset_sample(data_path: Path, sample_size: int = 10000):
    """Plot visualization for large datasets using sampling"""
    
    # Load data in chunks and sample
    sample_data = pd.read_csv(data_path, nrows=sample_size)
    
    plotter = IDSPlotter(FIGURES_DIR)
    plotter.plot_class_distribution(sample_data, save_name="sampled_distribution")
```

### Memory-Efficient Plotting

```python
def memory_efficient_plotting(data_path: Path):
    """Generate plots with minimal memory usage"""
    
    plotter = IDSPlotter(FIGURES_DIR)
    
    # Process data in chunks
    chunk_size = 5000
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        # Generate incremental plots
        plotter.plot_chunk_analysis(chunk)
        
        # Clear matplotlib cache
        plt.clf()
        plt.close('all')
```

## Troubleshooting

### Common Issues

1. **Memory Errors with Large Plots**
   ```python
   # Reduce DPI or use sampling
   plotter = IDSPlotter(FIGURES_DIR, dpi=150)  # Lower DPI
   
   # Or sample large datasets
   large_data = pd.read_csv(data_file)
   sample_data = large_data.sample(n=10000, random_state=42)
   ```

2. **Missing Font Issues**
   ```python
   # Use basic fonts
   plt.rcParams['font.family'] = 'sans-serif'
   plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
   ```

3. **Plot Not Saving**
   ```python
   # Ensure directory exists
   figures_dir = Path("./figures")
   figures_dir.mkdir(parents=True, exist_ok=True)
   
   # Check permissions
   import os
   print(f"Directory writable: {os.access(figures_dir, os.W_OK)}")
   ```

4. **Color Palette Issues**
   ```python
   # Reset to default seaborn palette
   import seaborn as sns
   sns.reset_defaults()
   sns.set_palette("husl")
   ```

## Integration Examples

### With Training Pipeline

```python
from rl_ids.modeling.train import DQNTrainer
from rl_ids.plots import IDSPlotter

class VisualizingTrainer(DQNTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = IDSPlotter()
    
    def on_episode_end(self, episode: int, metrics: dict):
        # Generate plots every 50 episodes
        if episode % 50 == 0:
            self.plotter.plot_training_metrics(
                self.metrics_file,
                save_name=f"training_episode_{episode}"
            )
```

### With API Service

```python
from fastapi import FastAPI
from rl_ids.plots import IDSPlotter

app = FastAPI()
plotter = IDSPlotter()

@app.get("/api/plots/confusion-matrix")
async def generate_confusion_matrix():
    """Generate and return confusion matrix plot"""
    
    plotter.plot_confusion_matrix(
        confusion_data,
        save_name="api_confusion_matrix"
    )
    
    return {"plot_url": "/figures/api_confusion_matrix.png"}
```

## See Also

- [Training Module](modeling.md) - Generate metrics data for plotting
- [Evaluation Module](modeling.md) - Generate evaluation data for analysis
- [Configuration Module](config.md) - Path configuration for figures
- [Getting Started Guide](../getting-started.md) - Basic plotting workflow
