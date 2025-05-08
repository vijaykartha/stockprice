import yfinance as yf

# Download historical price data from Yahoo Finance
ticker = "AAPL"  # Apple stock symbol
start_date = "2010-01-01"
end_date = "2024-01-01"
data = yf.download(ticker, start=start_date, end=end_date)

# Print the first few rows of data
print("\nFirst few rows of the data:")
print(data.head())

# Print information about the data
print("\nData information:")
print(data.info())

# Select closing prices
prices = data['Close'].values.reshape(-1, 1)