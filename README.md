# Stock Price Predictor

## Project Overview
This project explores whether simple machine learning signals can predict short-term stock price direction, and whether those predictions translate into a profitable trading strategy. The goal was to build something end-to-end: download historical stock data, engineer features, train a model, evaluate it, and then see how it performs as an actual trading strategy instead of just a prediction task.

## Why This Project?
I wanted to take machine learning concepts out of the classroom and actually apply them to a real problem. Financial markets felt like the perfect place to explore because they are noisy, unpredictable, and full of data.

I have always wondered whether code could help uncover patterns or signals in market data, even if they are small or imperfect. This project became a way to test that curiosity while learning through building.

## How It Works (Pipeline)
Right now the project follows this workflow:

1. Fetch historical NVDA stock data using `yfinance`
2. Engineer features:
   - daily returns  
   - 50-day and 200-day moving averages  
   - distance from price to each moving average  
3. Create a target label:
   - `1` if the next day closing price is higher  
   - `0` otherwise  
4. Train a logistic regression classifier  
5. Evaluate performance using:
   - accuracy  
   - ROC-AUC  
   - confusion matrix  
   - classification report  
6. Turn predictions into a trading strategy:
   - go long when the model predicts up  
   - stay in cash otherwise  
7. Backtest the strategy and compare it to buy-and-hold

## Tools and Libraries
- **Python**
- **yfinance** – fetch stock market data  
- **pandas & numpy** – data cleaning and manipulation  
- **matplotlib** – visualizations and trend exploration  
- **scikit-learn** – basic ML models (e.g., linear regression)  
- *Future scope:* PyTorch or TensorFlow for LSTM-based deep learning models  

## Learning Goals
- Work with real market data instead of toy datasets  
- Practice feature engineering and model evaluation  
- Learn how to structure and clean Python projects better  
- Understand the difference between prediction accuracy and real strategy performance  
- Develop an intuition for why many strategies fail in practice  

## Reflections and Takeaways
This project reminded me that building something from scratch is not about getting everything right the first time. It is about persistence, trial and error, and learning from mistakes along the way.

More specifically, I started by trying to predict whether a stock like NVDA would go up or down the next day. Through working on this, I learned a key lesson: daily price direction is extremely hard to predict.

The model’s predictions were basically no better than chance.

This helped me understand why real trading research usually focuses on:
- longer time horizons  
- richer and alternative features beyond simple price data  
- risk-adjusted performance instead of raw accuracy  

Now that I know this, my next step is to experiment with a weekly prediction horizon and see whether the signal becomes stronger when the time frame is less noisy.

## Next Steps
- Try predicting weekly instead of daily movement  
- Add more features such as volatility and rolling returns  
- Compare logistic regression to other models  
- Improve the backtest to make it more realistic  

## Cloud Extension (AWS)

This project includes a cloud-based extension:
- Input data stored in Amazon S3
- Preprocessing executed on an EC2 free-tier instance
- Outputs uploaded back to S3