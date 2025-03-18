# NBA Lineup Predictor (2007-2015)

A machine learning application that predicts the optimal fifth player for an NBA lineup based on historical matchup data from 2007-2015. This tool helps coaches and analysts make data-driven decisions about lineup combinations by analyzing player chemistry, position balance, and historical performance.

## Features

- Predict the best fifth player to complete a lineup
- Interactive web interface for easy predictions
- Visualize prediction confidence with detailed explanations
- View model evaluation metrics and performance statistics
- Account for player injuries and availability
- Season-specific predictions (2007-2015)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

Install all required packages using pip:

pip install -r requirements.txt


Required packages include:
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Project Structure

### Main Files:

- `app.py` - Flask web application entry point
- `src/model.py` - Core prediction model implementation
- `src/prediction_interface.py` - Interface for making predictions
- `src/data_preprocessing.py` - Data cleaning and preparation
- `src/chemistry_analysis.py` - Player chemistry analysis
- `src/position_features.py` - Position-based feature generation
- `src/time_analysis.py` - Time-based pattern analysis
- `analyze_test_results.py` - Script to evaluate model performance
- `templates/` - HTML templates for web interface

### Data Files:

- `data/matchups-{year}.csv` - Historical matchup data files for each season (2007-2015)
- `data/NBA_test.csv` - Test data for model evaluation
- `data/NBA_test_labels.csv` - Ground truth labels for test data

## Running the Application

### Web Interface

Start the Flask web server:

python app.py


Then visit http://127.0.0.1:5000/ in your web browser.

### Command Line Prediction

For CLI-based predictions:

python src/main.py


### Evaluating Model Performance

To analyze model performance on test data:

python analyze_test_results.py


## Model Evaluation

The model's performance can be evaluated through:

1. Web interface: Visit http://127.0.0.1:5000/evaluation
2. Running the evaluation script: `python analyze_test_results.py`

## Data Requirements

The application expects CSV files with the following naming convention:
- `data/matchups-{year}.csv` for years 2007-2015

Each file should contain columns for:
- `home_team` and `away_team`
- `home_0` through `home_4` (5 players for home team)
- `away_0` through `away_4` (5 players for away team)
- Additional statistics and game information

## Debugging

For debugging issues with test files:

python debug_test_files.py


## License

This project is intended for educational and research purposes.