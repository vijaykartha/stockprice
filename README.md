# Stock Price Prediction using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data from the Malaysian stock market.

## Features

- Data preprocessing and cleaning
- LSTM-based price prediction
- Model evaluation and visualization
- Support for multiple stocks
- Performance metrics calculation

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your stock market data in CSV format in the project directory
2. Run the data cleaning script:
```bash
python stock_analysis.py
```

3. Run the prediction model:
```bash
python stock_prediction.py
```

## Project Structure

- `stock_analysis.py`: Data cleaning and preprocessing
- `stock_prediction.py`: LSTM model implementation and training
- `requirements.txt`: Required Python packages
- `README.md`: Project documentation

## Model Architecture

The LSTM model consists of:
- 2 LSTM layers with 50 hidden units each
- Dropout layers (20%) for regularization
- Sequence length of 60 days
- Adam optimizer with learning rate 0.01

## Performance Metrics

The model is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 