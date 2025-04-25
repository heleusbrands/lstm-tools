"""
LSTM Model Training Example

This example demonstrates:
1. Loading time series data
2. Creating a dataset suitable for LSTM models
3. Building and training a simple LSTM model
4. Making predictions and visualizing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from lstm_tools import Sample
from lstm_tools.logger import configure_logging, info

# Configure logging
configure_logging(level=20)  # INFO level

# 1. Create synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
# Create a sine wave with some noise
x = np.linspace(0, 4 * np.pi, 500)
values1 = np.sin(x) + np.random.normal(0, 0.1, 500)  # Sine wave with noise
values2 = np.cos(x) + np.random.normal(0, 0.1, 500)  # Cosine wave with noise
values3 = np.cumsum(np.random.normal(0, 0.01, 500))  # Random walk

# Create a DataFrame
df = pd.DataFrame({
    'sine': values1,
    'cosine': values2,
    'random_walk': values3
}, index=dates)

print("Created synthetic DataFrame:")
print(df.head())

# 2. Create a Sample from the DataFrame
sample = Sample.from_dataframe(df)
print(f"Created Sample with {len(sample)} timeframes and features: {sample.feature_names}")

# 3. Create a dataset suitable for LSTM with a lookback of 30 days to predict 7 days ahead
lookback = 30
forecast = 7
batch_size = 16

X, y, X_time, y_time = sample.create_lstm_dataset(
    target_feature='sine',  # We're predicting only the sine wave
    lookback=lookback,
    forecast=forecast,
    batch_size=batch_size
)

print(f"Created dataset: X shape: {X.shape}, y shape: {y.shape}")

# 3.5 Demonstrate Chronicle batch compression 
print("\nDemonstrating Chronicle batch compression:")
# Create historical windows for feature extraction
historical_data = sample.historical_sliding_window()
# Apply batch compression to extract multiple features
compressed_features = historical_data.batch_compress(
    features=['sine', 'cosine'],  # Only compress sine and cosine features
    methods={
        'mean': np.mean,
        'std': np.std,
        'range': lambda x: np.max(x) - np.min(x)
    }
)

# Print some of the compressed results
print(f"Compressed features shape examples:")
for key, value in list(compressed_features.items())[:3]:  # Show first 3 results
    print(f"  {key}: {value.shape} - Sample value: {value[0]:.4f}")

# 4. Create DataLoader for PyTorch
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. Define a simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # We only need the last time step's output
        output = self.fc(lstm_out[:, -1, :])
        return output

# 6. Initialize model, loss function, and optimizer
input_size = X.shape[2]  # Number of features
hidden_size = 50
output_size = y.shape[1]  # Number of time steps to predict

model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Print training statistics
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

# 8. Evaluate the model
model.eval()
test_loss = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        
        all_predictions.append(outputs)
        all_targets.append(batch_y)

print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# 9. Visualize the results for the first test sample
predictions = torch.cat(all_predictions, dim=0).numpy()
targets = torch.cat(all_targets, dim=0).numpy()

plt.figure(figsize=(12, 6))
plt.plot(range(forecast), targets[0], 'b-', label='Actual')
plt.plot(range(forecast), predictions[0], 'r--', label='Predicted')
plt.title('LSTM Prediction vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.savefig("lstm_prediction.png")
plt.show()

print("\nModel training complete and results saved to lstm_prediction.png") 