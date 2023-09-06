import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the CNN model
model = keras.Sequential([
    Conv2D(4, (3, 3), activation='gelu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(8, (3, 3), activation='gelu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    
    Dense(32, activation='gelu'),
    
    Dropout(0.5),
   
    Dense(10, activation='softmax')
])

def evaluate(model, x_test, y_test):
    # Evaluate the model on the test data
    y_pred = model.predict(x_test.reshape(-1, 28, 28, 1))
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate the confusion matrix
    confusion_mtx = confusion_matrix(y_test, y_pred_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
    print(f'Test accuracy: {test_acc}')


if __name__ == "__main__":
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train the model
    model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5, validation_split=0.2)
    
    evaluate(model, x_test, y_test)