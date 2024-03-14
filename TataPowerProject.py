!pip install scikit-learn

%tensorflow_version 2.x

from _future_ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

path1="/content/drive/MyDrive/ML TATA power/MLTATAPOWER_TRAIN.csv"
path2="/content/drive/MyDrive/ML TATA power/ML EVAL.csv"

# Loading dataset.
train_df = pd.read_csv(path1) # training data variable
predict_df = pd.read_csv(path2) # testing data variable

train_df.columns = train_df.columns.str.replace(' ', '_')
predict_df.columns = predict_df.columns.str.replace(' ', '_')

train = train_df.copy();
test = predict_df.copy();

# Defining features and target variable for training
X_train = train_df[['FLU_GAS_TEMP', 'STEAM_PRESSURE', 'TURBINE_SPEED', 'GENERATOR_VOLTAGE', 'AIR_QUALITY']]
y_train = train_df['HEALTHINESS']

# Standardizing features for training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Building the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Defining features for prediction
X_predict = predict_df[['FLU_GAS_TEMP', 'STEAM_PRESSURE', 'TURBINE_SPEED', 'GENERATOR_VOLTAGE', 'AIR_QUALITY']]

# Standardizing features for prediction
X_predict = scaler.transform(X_predict)

# Making predictions on the prediction dataset
predicted_probabilities = model.predict(X_predict)

i=2

# Printing the predicted probabilities for the prediction dataset
print("\nPredicted Healthiness percentage for ",i," index of the Evaluating dataset : ",predicted_probabilities[i]*100,"%")

# Error description
error_desc = ['NO ERROR', 'ERROR A', 'ERROR B', 'ERROR C']

# Defining features and target variable for training (for ERROR_TYPE)
X_train_error = train_df[['FLU_GAS_TEMP', 'STEAM_PRESSURE', 'TURBINE_SPEED', 'GENERATOR_VOLTAGE', 'AIR_QUALITY']]
y_train_error = train_df['ERROR_TYPE'].astype(int)  # Ensure labels are integers

# Standardizing features for training (for ERROR_TYPE)
scaler_error = StandardScaler()
X_train_error = scaler_error.fit_transform(X_train_error)

# Splitting the dataset into training and validation sets (for ERROR_TYPE)
X_train_error, X_val_error, y_train_error, y_val_error = train_test_split(
    X_train_error, y_train_error, test_size=0.2, random_state=42
)

# Building the DNN model for ERROR_TYPE using tf.keras
model_error = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_error.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # Assuming 4 classes for ERROR_TYPE
])

# Compile the model
model_error.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_error.fit(X_train_error, y_train_error, epochs=10, batch_size=32, validation_data=(X_val_error, y_val_error))

# Defining features for prediction (for ERROR_TYPE)
X_predict_error = predict_df[['FLU_GAS_TEMP', 'STEAM_PRESSURE', 'TURBINE_SPEED', 'GENERATOR_VOLTAGE', 'AIR_QUALITY']]

# Standardizing features for prediction (for ERROR_TYPE)
X_predict_error = scaler_error.transform(X_predict_error)

# Making predictions on the prediction dataset (for ERROR_TYPE)
predicted_probabilities_error = model_error.predict(X_predict_error)

# Extracting predicted classes
predicted_classes_error = np.argmax(predicted_probabilities_error, axis=1)

# Printing the predicted classes for the prediction dataset (for ERROR_TYPE)
print("\nThe error type for index ",i," of the evaluating dataset is: ",error_desc[predicted_classes_error[i]])