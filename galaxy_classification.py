# Import necessary libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from importData import load_galaxy_data
from PIL import Image

def preprocess_data(dataset):
    # Resize images to a smaller size to reduce memory usage
    new_size = (64, 64)
    images = np.array([np.array(Image.fromarray(np.array(image['image'])).resize(new_size)) for image in dataset['train']])
    labels = np.array(dataset['train']['label'])
    # Normalize the images
    images = images / 255.0
    # Split the data
    return train_test_split(images, labels, test_size=0.2, random_state=42)

def build_model(layer1_neuron, layer2_neuron, layer3_neuron, layer4_neuron, lr, kernel_size, activation_function, dropout_value):
    # Build a convolutional neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(layer1_neuron, kernel_size, activation=activation_function),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(layer2_neuron, kernel_size, activation=activation_function),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(layer3_neuron, kernel_size, activation=activation_function),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(layer4_neuron, activation=activation_function),
        tf.keras.layers.Dropout(dropout_value),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Parameters
nb_epochs = 15 #15
layer1_neuron = 32 #32
layer2_neuron = 64 #64
layer3_neuron = 128 #128
layer4_neuron = 256 #128
lr = 0.001 #learning_rate = 0.001
kernel_size = (3, 3) #(3, 3)
activation_function = 'relu' #'relu'
dropout_value = 0.5 #0.5

# Loading Dataset
dataset = load_galaxy_data()
train_images, test_images, train_labels, test_labels = preprocess_data(dataset)

# Training and Testing neural network
model = build_model(layer1_neuron, layer2_neuron, layer3_neuron, layer4_neuron, lr, kernel_size, activation_function, dropout_value)
model.fit(train_images, train_labels, epochs=nb_epochs, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")