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

def build_model(layer1_neuron, layer2_neuron, layer3_neuron, layer4_neuron, lr):
    # Build a convolutional neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(layer1_neuron, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(layer2_neuron, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(layer3_neuron, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(layer4_neuron, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

nb_epochs = 15 #15
layer1_neuron = 32 #32
layer2_neuron = 64 #128
layer3_neuron = 128 #64
layer4_neuron = 128
lr = 0.001 #learning_rate = 0.001

dataset = load_galaxy_data()
train_images, test_images, train_labels, test_labels = preprocess_data(dataset)
model = build_model(layer1_neuron, layer2_neuron, layer3_neuron, layer4_neuron, lr)
model.fit(train_images, train_labels, epochs=nb_epochs, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")