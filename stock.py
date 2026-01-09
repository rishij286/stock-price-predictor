import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend for matplotlib

import os # ensure output directory exists

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    recall_score, 
    classification_report, 
    roc_auc_score, 
    roc_curve,
)


plt.style.use('seaborn-v0_8-whitegrid')

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# download stock data function, return cleaned dataframe
def load_data(ticker: str = "NVDA", period: str = "5y") -> pd.DataFrame:
    # Try yfinance first
    df = yf.download([ticker], period=period, progress=False)

    # If yfinance fails/empty, fall back to local CSV
    if df is None or df.empty:
        stock_csv = f"{ticker}_data.csv"
        print(f"yfinance download failed/empty. Falling back to {stock_csv}")

        df = pd.read_csv(stock_csv)

        # Handle your "Price/Close/High/Low/Open/Volume" CSV format
        if "Price" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Price": "Date"})
        if "Close" in df.columns and "Close_NVDA" not in df.columns:
            df = df.rename(columns={"Close": "Close_NVDA"})

        # Parse Date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).reset_index(drop=True)
        else:
            raise KeyError(f"No Date column found in {stock_csv}. Columns: {df.columns.tolist()}")

        # Make Close numeric
        df["Close_NVDA"] = (
            df["Close_NVDA"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df["Close_NVDA"] = pd.to_numeric(df["Close_NVDA"], errors="coerce")
        df = df.dropna(subset=["Close_NVDA"]).reset_index(drop=True)

        # Add Year/Month
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        return df

    # Save downloaded data
    stock_csv = f"{ticker}_data.csv"
    df.to_csv(stock_csv)
    print(f"Saved downloaded data to {stock_csv}")

    # flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

    # ensure close is named correctly
    close_col = "Close_NVDA" if "Close_NVDA" in df.columns else "Close"
    df = df.rename(columns={close_col: "Close_NVDA"})

    df = df.reset_index()

    # ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    return df

# add returns, moving averages, and target variable
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # daily return
    df['Daily_Return'] = df['Close_NVDA'].pct_change()

    # previous day's close
    df['Prev_Close'] = df['Close_NVDA'].shift(1)

    # target variable: 1 if next day's close is higher, else 0
    df['Target'] = (df['Close_NVDA'].shift(-1) > df['Close_NVDA']).astype(int)

    """
    Moving Averages: Avg closing price over a certain peroid of time
    -------------------------------------------------------------------
    Short-term moving averages crossing above longer-term moving averages is 
    generally seen as bullish and short-term moving averages crossing from above
    to below long-term moving averages is generally seen as bearish.
    """

    df['MA_50'] = df['Close_NVDA'].rolling(window=50).mean()
    df['MA_200'] = df['Close_NVDA'].rolling(window=200).mean()

    df['Close_vs_MA_50'] = df['Close_NVDA'] - df['MA_50']
    df['Close_vs_MA_200'] = df['Close_NVDA'] - df['MA_200']

    """
    Moving Average difference percentage:
    if it >= 0.1 (10%), bullish, strong uptrend
    if it <= -0.05 (-5%), bearish, moderate downtrend
    if it ~ 0, neutral
    """

    df['MA_Diff_Pct'] = (df['MA_50'] - df['MA_200']) / df['MA_200']

    """
    Plotting The MA Difference Percentage, recall:
    - if it >= 0.1 (10%), bullish, strong uptrend
    - if it <= -0.05 (-5%), bearish, moderate downtrend
    - if it ~ 0, neutral

    Also plot the closing price, and both moving averages along with it for
    comparison
    """

    df = df.dropna().reset_index(drop=True)
    return df

# plot 50 and 200 day moving averages along with their difference percentage
def plot_moving_averages(df: pd.DataFrame, outpath="outputs/moving_averages.png") -> None:
    fig, axs = plt.subplots(2, figsize=(10, 6))
    fig.suptitle("Price & Moving Avg VS Difference in MAs")

    axs[0].plot(df["Date"], df["Close_NVDA"])
    axs[0].plot(df["Date"], df["MA_50"])
    axs[0].plot(df["Date"], df["MA_200"])
    axs[0].legend(["Close", "MA_50", "MA_200"])
    axs[0].set_ylabel("Price ($)")

    axs[1].plot(df["Date"], df["MA_Diff_Pct"])
    axs[1].set_ylabel("MA Diff %")
    axs[1].axhline(0, color="black", linestyle="--")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved {outpath}")



# prepare the data for modeling
def prepare_data(df: pd.DataFrame):
    features = ['MA_50', 'MA_200', 'MA_Diff_Pct', 'Prev_Close', 'Daily_Return']

    X = df[features].values   # converts pandas DataFrame -> numpy array
    y = df['Target'].values   # converts Series -> numpy array

    #Now use the train test split library
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # test size = 0.2 means 20% of data goes to testing, and 80% to training
    # shuffle false is needed for time series data like stocks

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    return model



#evaluate the predictions
def evaluate_model(model, X_test, y_test, y_pred, df: pd.DataFrame):
    #Simple evaluation, accuracy:
    print(f"Simple Prediction: {accuracy_score(y_test, y_pred):.3f}")

    #now compare this to baselines: such as always UP or always DOWN
    always_up_acc = (y_test == 1).mean()
    print(f"Baseline, always up accuracy: {always_up_acc:.3f}")

    #Check the distribution
    y = df['Target'].values
    print(f"Proportion of up days: {y.mean():.3f}")

    #Now check with confusion matrix
    print(confusion_matrix(y_test, y_pred))

    #Check with other metric other than accuracy, using recall first:
    specificity = recall_score(y_test, y_pred, pos_label=0)
    print(f"Specificity (True Negative Rate): {specificity:.2f}")

    #Check the models certainty in guessing UP:
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Lowest probability of UP: {y_proba.min():.3f}")
    print(f"Highest probability of UP: {y_proba.max():.3f}")
    print(f"Mean of UP predictions: {y_proba.mean():.3f}")

    # plot roc-auc and precision-recall curves
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.2f}")

    #tpr = true positive rate
    #fpr = false positive rate
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png") # save ROC curve plot
    plt.close()
    print("Saved outputs/roc_curve.png")

    #Classification report for more detailed metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# time to shift a little bit, as results are random.
# turn it into a trading strategy and see if it works out better that way.

def backtest_daily_strategy(df: pd.DataFrame, y_pred, train_len: int):
    # 1) align test window with nvda rows
    test_start = train_len # index where test set starts
    test_end = test_start + len(y_pred)
    test_df = df.iloc[test_start:test_end].copy()

    # 2) realized return for each decision
    # for each day t, your label/decision is about the move from t to t+1
    # which is daily_return at t+1
    test_df["Next_Return"] = df["Daily_Return"].shift(-1).iloc[test_start:test_end].values

    # drop rows without Next_Return
    test_df = test_df.dropna(subset=["Next_Return"])

    # 3) trading signal from model predictions (1=long, 0=cash)
    signals = pd.Series(y_pred[: len(test_df)], index=test_df.index, name="Signal")

    # 4) strategy returns vs buy and hold
    # strategy: only earn return when signal is 1 (long)
    test_df["Strategy_Return"] = signals * test_df["Next_Return"]
    # buy and hold: always earn the return
    test_df["Buy_and_Hold_Return"] = test_df["Next_Return"]

    # cumulative returns
    test_df["Strategy_Equity"] = (1 + test_df["Strategy_Return"]).cumprod()
    test_df["Buy_and_Hold_Equity"] = (1 + test_df["Buy_and_Hold_Return"]).cumprod()

    # print total returns over the test period
    total_strategy_return = test_df["Strategy_Equity"].iloc[-1] - 1
    total_bh_return = test_df["Buy_and_Hold_Equity"].iloc[-1] - 1

    print(f"Total Strategy Return: {total_strategy_return:.2%}")
    print(f"Total Buy and Hold Return: {total_bh_return:.2%}")

    # 7) Simple Sharpe ratio estimate
    trading_days_per_year = 252
    strategy_ann_return = test_df["Strategy_Return"].mean() * trading_days_per_year
    strategy_ann_vol = test_df["Strategy_Return"].std() * np.sqrt(trading_days_per_year)

    if strategy_ann_vol > 0:
        strategy_sharpe = strategy_ann_return / strategy_ann_vol
        print(f"Strategy Sharpe (approx): {strategy_sharpe:.2f}")
    else:
        print("Strategy Sharpe: undefined (zero volatility)")

    # plot equity curves
    plt.figure(figsize=(10, 5))
    plt.plot(test_df["Date"], test_df["Strategy_Equity"], label="Strategy")
    plt.plot(test_df["Date"], test_df["Buy_and_Hold_Equity"], label="Buy & Hold")
    plt.title("Strategy vs Buy & Hold (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Equity (starting at 1.0)")
    plt.legend()
    plt.tight_layout()

    # save the equity curve plot
    plt.savefig("outputs/equity_curve.png")
    plt.close()
    print("Saved outputs/equity_curve.png")

# save processed data to CSV
def save_data(df: pd.DataFrame, filename: str = "outputs/processed_data.csv") -> None:
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    # ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # load and preprocess data
    df = load_data()
    df = add_features(df)

    # plot moving averages and their difference percentage
    plot_moving_averages(df)

    # prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # evaluate model performance using various metrics
    evaluate_model(model, X_test, y_test, y_pred, df)
    backtest_daily_strategy(df, y_pred, len(X_train))
    save_data(df)


if __name__ == "__main__":
    main()