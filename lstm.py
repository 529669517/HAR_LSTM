import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

np.random.seed(42)
tf.random.set_seed(42)

SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
    ]


def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        filename = f'UCI_HAR_Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'
        data = pd.read_csv(filename, delim_whitespace=True, header=None)
        signals_data.append(data.to_numpy())

    return np.transpose(signals_data, (1, 2, 0))


def load_label(subset):
    filename = f'UCI_HAR_Dataset/{subset}/y_{subset}.txt'
    y = pd.read_csv(filename, delim_whitespace=True, header=None)[0]

    return pd.get_dummies(y).to_numpy()


def load_data():
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_label('train'), load_label('test')

    return X_train, X_test, y_train, y_test


# Initializing parameters
n_hidden = 64

# Loading the train and test data
X_train, X_test, Y_train, Y_test = load_data()


time_steps = len(X_train[0])
input_dim = len(X_train[0][0])
# walking, upstairs, downstairs, laying, standing, sitting
n_classes = 6


# Initializing the sequential model
model = Sequential()
# Configuring the parameters
model.add(LSTM(n_hidden, input_shape=(time_steps, input_dim), name='input'))
# Adding a dropout layer
model.add(Dropout(0.5))
# Adding a dense output layer with sigmoid activation
model.add(Dense(n_classes, activation='softmax', name='output'))
model.summary()


# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# prepare callbacks
callbacks = [ModelCheckpoint('lstm.h5', save_weights_only=False, save_best_only=True, verbose=1)]

# Training the model
model.fit(X_train, Y_train, batch_size=16, validation_data=(X_test, Y_test), epochs=30, callbacks=callbacks)


# The steps for creating the confusion matrix
y_pred_one = model.predict(X_test)

y_pred_labels = np.argmax(y_pred_one, axis=1)
y_true_labels = np.argmax(Y_test, axis=1)

confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)

LABELS = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying', ]

plt.figure(figsize=(12, 10))
sns.set(style='darkgrid', font_scale=1.5)
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix of LSTM")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# save function
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 128, 9], model.inputs[0].dtype))


MODEL_DIR = "saved_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

# Save the model.
with open('lstm.tflite', 'wb') as f:
  f.write(tflite_model)

# check the input and output shape of the tf_lite model
interpreter = tf.lite.Interpreter(model_path="lstm.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

