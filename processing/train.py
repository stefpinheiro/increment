import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load your dataset and prepare it
# X_train, X_val, y_train, y_val = ...

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

# Save metrics to a CSV file
metrics = pd.DataFrame(history.history)
metrics.to_csv('metrics.csv', index=False)
