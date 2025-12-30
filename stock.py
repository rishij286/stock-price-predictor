import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


plt.style.use('seaborn-v0_8-whitegrid')

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)


# Download NVDA data
nvda = yf.download(["NVDA"], period='5y')

# Flatten MultiIndex columns if needed
if isinstance(nvda.columns, pd.MultiIndex):
    nvda.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in nvda.columns]

# Reset index to bring 'Date' into columns
nvda = nvda.reset_index()

# Extract Year and Month
nvda['Year'] = nvda['Date'].dt.year
nvda['Month'] = nvda['Date'].dt.month

#now add a daily return part:
nvda['Daily_Return'] = nvda['Close_NVDA'].pct_change()

# the previous days' close
nvda['Prev_Close'] = nvda['Close_NVDA'].shift(1)

# int val (1 / 0) for if price went up or not
nvda['Target'] = (nvda['Close_NVDA'].shift(-5) > nvda['Close_NVDA']).astype(int)

"""
Moving Averages: Avg closing price over a certain peroid of time
-------------------------------------------------------------------
Short-term moving averages crossing above longer-term moving averages is 
generally seen as bullish and short-term moving averages crossing from above
to below long-term moving averages is generally seen as bearish.
"""
nvda['MA_50'] = nvda['Close_NVDA'].rolling(window=50).mean()
nvda['MA_200'] = nvda['Close_NVDA'].rolling(window=200).mean()

nvda['Close_vs_MA_50'] = nvda['Close_NVDA'] - nvda['MA_50']
nvda['Close_vs_MA_200'] = nvda['Close_NVDA'] - nvda['MA_200']

"""
Moving Average difference percentage:
if it >= 0.1 (10%), bullish, strong uptrend
if it <= -0.05 (-5%), bearish, moderate downtrend
if it ~ 0, neutral
"""
nvda['MA_Diff_Pct'] = (nvda['MA_50'] - nvda['MA_200']) / nvda['MA_200']

"""
Plotting The MA Difference Percentage, recall:
- if it >= 0.1 (10%), bullish, strong uptrend
- if it <= -0.05 (-5%), bearish, moderate downtrend
- if it ~ 0, neutral

Also plot the closing price, and both moving averages along with it for
comparison
"""
fig, axs = plt.subplots(2)
fig.suptitle('Price & Moving Avg VS Difference in MAs')
axs[0].plot(nvda['Date'], nvda['Close_NVDA'])
axs[0].plot(nvda['Date'], nvda['MA_50'])
axs[0].plot(nvda['Date'], nvda['MA_200'])
axs[0].legend(["Close", "MA_50", "MA_200"])
axs[0].set_ylabel("Price ($)")

axs[1].plot(nvda['Date'], nvda['MA_Diff_Pct'])
axs[1].set_ylabel("MA Diff %")
axs[1].axhline(0, color="black", linestyle='--')
plt.show()

# drop rows with NaN
nvda = nvda.dropna()


# Now onto training the model.

#Initializing features X and Y:

# Example feature set
features = ['MA_50', 'MA_200', 'MA_Diff_Pct', 'Prev_Close', 'Daily_Return']

X = nvda[features].values   # converts pandas DataFrame -> numpy array
y = nvda['Target'].values   # converts Series -> numpy array

#Now use the train test split library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# test size = 0.2 means 20% of data goes to testing, and 80% to training
# shuffle false is needed for time series data like stocks

#initialize the logistic regression
logreg = LogisticRegression(class_weight='balanced')

#fit and predict
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#evaluate the predictions
#Simple evaluation, accuracy:
print(f"Simple Prediction: {accuracy_score(y_test, y_pred)}")

#now compare this to baselines: such as always UP or always DOWN
total_up = nvda['Target'].sum()
total_rows = len(nvda)
always_up_acc = total_up / total_rows
print(always_up_acc)



#now do the actual thing with accuracy score to compare:
y_pred_baseline = np.ones_like(y_test)

baseline_acc = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Prediction, always up: {baseline_acc}")

#The prediction and baseline are the exact same thing.. we may need to rethink Logistic Regression.
#First, lets run some diagnostics to see what our issue is.

#Check the distribution
print(f"Proportion of up days: {y.mean()}")

#Now check with confusion matrix
print(confusion_matrix(y_test, y_pred))

#Check with other metric other than accuracy, using recall first:
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity (True Negative Rate): {specificity:.2f}")

#Check the models certainty in guessing UP:
y_proba = logreg.predict_proba(X_test)[:, 1]

print("Lowest probability of UP: ", y_proba.min())
print("Highest probability of UP: ", y_proba.max())
print("Mean of UP predictions:", y_proba.mean())

# plot roc-auc and precision-recall curves
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc:.2f}")

#tpr = true positive rate
#fpr = false positive rate
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.show()

#Classification report for more detailed metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# time to shift a little bit, as results are random.
# turn it into a trading strategy and see if it works out better that way.

# 1) align test window with nvda rows
test_start = len(X_train) # index where test set starts
test_end = test_start + len(y_test)
test_nvda = nvda.iloc[test_start:test_end].copy()

# 2) realized return for each decision
# for each day t, your label/decision is about the move from t to t+1
# which is daily_return at t+1
test_nvda["Next_Return"] = nvda["Daily_Return"].shift(-1).iloc[test_start:test_end].values

# drop rows without Next_Return
test_nvda = test_nvda.dropna(subset=["Next_Return"])

# 3) trading signal from model predictions (1=long, 0=cash)
signals = pd.Series(y_pred[: len(test_nvda)], index=test_nvda.index, name="Signal")

# 4) strategy returns vs buy and hold
# strategy: only earn return when signal is 1 (long)
test_nvda["Strategy_Return"] = signals * test_nvda["Next_Return"]

# buy and hold: always earn the return
test_nvda["Buy_and_Hold_Return"] = test_nvda["Next_Return"]

# cumulative returns
test_nvda["Strategy_Equity"] = (1 + test_nvda["Strategy_Return"]).cumprod()
test_nvda["Buy_and_Hold_Equity"] = (1 + test_nvda["Buy_and_Hold_Return"]).cumprod()

# print total returns over the test period
total_strategy_return = test_nvda["Strategy_Equity"].iloc[-1] - 1
total_bh_return = test_nvda["Buy_and_Hold_Equity"].iloc[-1] - 1

print(f"Total Strategy Return: {total_strategy_return:.2%}")
print(f"Total Buy and Hold Return: {total_bh_return:.2%}")

# 7) Simple Sharpe ratio estimate (optional, but nice)
trading_days_per_year = 252
strategy_ann_return = test_nvda["Strategy_Return"].mean() * trading_days_per_year
strategy_ann_vol = test_nvda["Strategy_Return"].std() * np.sqrt(trading_days_per_year)

if strategy_ann_vol > 0:
    strategy_sharpe = strategy_ann_return / strategy_ann_vol
    print(f"Strategy Sharpe (approx): {strategy_sharpe:.2f}")
else:
    print("Strategy Sharpe: undefined (zero volatility)")

# plot equity curves
plt.figure(figsize=(10, 5))
plt.plot(test_nvda["Date"], test_nvda["Strategy_Equity"], label="Strategy")
plt.plot(test_nvda["Date"], test_nvda["Buy_and_Hold_Equity"], label="Buy & Hold")
plt.title("Strategy vs Buy & Hold (Test Period)")
plt.xlabel("Date")
plt.ylabel("Equity (starting at 1.0)")
plt.legend()
plt.tight_layout()
plt.show()