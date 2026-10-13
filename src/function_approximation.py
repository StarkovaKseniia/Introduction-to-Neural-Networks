import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Step 1 & 2: Define Function & Generate Training Data
np.random.seed(42)
x = np.linspace(-10, 10, 2000)
# Ground truth function with a small amount of simulated noise
y = np.sin(x) + 0.1 * (x ** 2) + np.random.normal(0, 0.05, size=x.shape)

# Reshape x for Keras (expects 2D array: samples x features)
X = x.reshape(-1, 1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create Neural Network Model (MLP)
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=0
)

# Step 5: Evaluate the Model
val_loss = model.evaluate(X_val, y_val)
print(f"Validation MSE Loss: {val_loss:.4f}")

# Step 6: Make Predictions & Visualize
x_test = np.linspace(-10, 10, 500).reshape(-1, 1)
y_true = np.sin(x_test) + 0.1 * (x_test ** 2)
y_pred = model.predict(x_test)

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, label='True Function: sin(x) + 0.1x^2', color='blue', linewidth=2)
plt.plot(x_test, y_pred, label='Neural Network Prediction', color='red', linestyle='--', linewidth=2)
plt.scatter(X_train, y_train, color='gray', alpha=0.3, label='Training Data (with noise)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Approximation using a Feedforward Neural Network')
plt.legend()
plt.show()