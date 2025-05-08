import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(df, target_column='Close', sequence_length=60):
    """
    Prepare data for LSTM model
    """
    # Select the target column
    data = df[target_column].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences for training
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """
    Train the LSTM model
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

def plot_predictions(actual, predicted, title, dates=None):
    """
    Plot actual vs predicted values with dates on X-axis
    """
    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, actual, label='Actual')
        plt.plot(dates, predicted, label='Predicted')
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    else:
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title}.png')
    plt.close()

def evaluate_predictions(actual, predicted):
    """
    Calculate and print evaluation metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the cleaned data
    print("Loading data...")
    df = pd.read_csv('cleaned_stock_market.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort data by date
    df = df.sort_values('Date')
    
    # Select a specific stock (e.g., first stock in the dataset)
    stock_ticker = df['Ticker'].iloc[0]
    print(f"\nAnalyzing stock: {stock_ticker}")
    
    # Filter data for the selected stock
    stock_data = df[df['Ticker'] == stock_ticker].copy()
    
    # Set sequence length for LSTM
    sequence_length = 60
    
    # Prepare data
    X, y, scaler = prepare_data(stock_data, sequence_length=sequence_length)
    
    # Get dates for plotting
    dates = stock_data['Date'].values[sequence_length:]
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    # Split data into training and testing sets (80% training, 20% testing)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create and train the model
    print("\nCreating and training LSTM model...")
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train.to(device)).cpu().numpy()
        test_predictions = model(X_test.to(device)).cpu().numpy()
    
    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.numpy())
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    
    # Plot training results
    plot_predictions(
        y_train_actual,
        train_predictions,
        f'Training Predictions - {stock_ticker}',
        dates_train
    )
    
    # Plot testing results
    plot_predictions(
        y_test_actual,
        test_predictions,
        f'Testing Predictions - {stock_ticker}',
        dates_test
    )
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    evaluate_predictions(y_test_actual, test_predictions)
    
    # Save the model
    torch.save(model.state_dict(), f'lstm_model_{stock_ticker}.pth')
    print(f"\nModel saved as 'lstm_model_{stock_ticker}.pth'")

if __name__ == "__main__":
    main() 