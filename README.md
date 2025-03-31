# Stock Price Prediction with Deep Learning and XGBoost

This project implements and compares machine learning approaches for stock price prediction, focusing on LSTM (Long Short-Term Memory) neural networks and XGBoost. The system analyzes historical stock data to predict future price movements.

## Project Overview

This project aims to predict stock prices using:
- **LSTM Neural Networks**: To capture temporal dependencies in time-series data
- **XGBoost**: A powerful gradient boosting framework for comparison
- **Technical Indicators**: Including moving averages, MACD, RSI, and Bollinger Bands

Based on our analysis, LSTM significantly outperforms XGBoost for this task, with the LSTM model achieving an R² score of 0.9524 compared to XGBoost's -2.0623.

## Features

- Historical stock data retrieval via Yahoo Finance
- Comprehensive feature engineering of technical indicators
- Data preprocessing and normalization
- Implementation of LSTM and XGBoost models
- Model performance comparison and visualization
- Future price prediction functionality

## Performance Metrics

| Model   | MSE      | RMSE    | MAE     | R²       |
|---------|----------|---------|---------|----------|
| LSTM    | 20.8419  | 4.5653  | 3.5478  | 0.9524   |
| XGBoost | 1339.8199| 36.6036 | 32.4474 | -2.0623  |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-prediction-deep-learning-xgboost.git
cd stock-prediction-deep-learning-xgboost

# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
xgboost>=1.5.0
yfinance>=0.1.70
seaborn>=0.11.0
```

## Usage

### Basic Usage

```python
# Run the main prediction script
python predict_stock.py --ticker AAPL --days 5
```

### Customize Your Analysis

```python
# Train with custom parameters
python train_models.py --ticker MSFT --start_date 2018-01-01 --end_date 2023-01-01 --lstm_units 100 --xgb_depth 8
```

## Project Structure

```
stock-prediction/
│
├── data/                  # Data storage
├── models/                # Saved model files
├── notebooks/             # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   └── model_comparison.ipynb
├── src/                   # Source code
│   ├── __init__.py
│   ├── data_processing.py # Data retrieval and preprocessing
│   ├── feature_eng.py     # Feature engineering
│   ├── lstm_model.py      # LSTM model implementation
│   ├── xgboost_model.py   # XGBoost model implementation
│   └── utils.py           # Utility functions
├── tests/                 # Unit tests
├── predict_stock.py       # Main prediction script
├── train_models.py        # Model training script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Key Findings

Our analysis revealed significant performance differences between LSTM and XGBoost models:

1. **LSTM Superiority**: The LSTM model demonstrated excellent performance with an R² of 0.9524, explaining about 95% of the variance in stock prices.

2. **XGBoost Limitations**: XGBoost showed poor performance with a negative R² value, suggesting challenges in capturing the temporal patterns critical for stock prediction.

3. **Error Measurements**: LSTM achieved an average prediction error (MAE) of approximately $3.55, while XGBoost's error was about $32.45.

These findings suggest that temporal dependencies in stock price data are crucial, which LSTM is specifically designed to capture.

## Future Work

- Implement hyperparameter tuning for both models
- Add sentiment analysis from financial news and social media
- Explore additional deep learning architectures (GRU, CNN-LSTM)
- Develop ensemble methods that combine multiple models
- Incorporate additional market indicators and macroeconomic factors
- Add backtesting functionality to simulate real trading scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
