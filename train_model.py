import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data (with cluster labels from the previous step)
df = pd.read_csv("tracks_with_clusters.csv")

feature_columns = ["tempo", "energy", "brightness", "noisiness", "tonality"]
X = df[feature_columns].values
y = df["cluster"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
# NOTE: with only 35 samples, this split is very small - expected for a first version
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

num_classes = len(np.unique(y))

# Define a simple feedforward neural network
class MusicMoodClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = MusicMoodClassifier(input_size=len(feature_columns), num_classes=num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    predicted = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted == y_test_t).float().mean().item()

print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print(f"(Note: with only {len(df)} samples total, this accuracy is not very meaningful yet.")
print("More data would make this a much more reliable model.)")

# Save the trained model
torch.save(model.state_dict(), "music_mood_model.pth")
print("\nModel saved to music_mood_model.pth")