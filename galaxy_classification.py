# Import necessary libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from importData import load_galaxy_data
from PIL import Image
from datetime import datetime

def preprocess_data(dataset):
    new_size = (64, 64)
    images = np.array([np.array(Image.fromarray(np.array(image['image'])).resize(new_size)) for image in dataset['train']])
    labels = np.array(dataset['train']['label'])

    images = images / 255.0

    return train_test_split(images, labels, test_size=0.2, random_state=42)

# Build a convolutional neural network model
def build_model(layer1_neuron, layer2_neuron, layer3_neuron, layer4_neuron, lr, kernel_size, activation_function, dropout_value, opt_name):
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
    model.compile(optimizer=opt_name(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Parameters
nb_epochs = 15 #15
layer1_neuron = 32 #32
layer2_neuron = 64 #64
layer3_neuron = 128 #128
layer4_neuron = 64 #64

kernel_size = (3, 3) #(3, 3)
activation_function = 'relu' #'relu'
dropout_value = 0.5 #0.5

optimizers = {
    'SGD': tf.keras.optimizers.SGD,
    'Adagrad': tf.keras.optimizers.Adagrad,
    'Adam': tf.keras.optimizers.Adam
}

learning_rates = [1e-3, 1e-2, 1e-1]

print("\nLoading Dataset\n")
dataset = load_galaxy_data()
train_images, test_images, train_labels, test_labels = preprocess_data(dataset)

best_opt = ''
best_lr = 0
best_acc = 0

print("\nIterate on each optimizer and each learning_rate\n")
for opt_name, opt_class in optimizers.items():
    for lr in learning_rates:
        # Create a logs directory
        log_dir = f"runs/{opt_name}_lr_{lr}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        print(f"\nCreated {log_dir} file\n")

        # Training and Testing neural network
        print("\nTraining...\n")
        model = build_model(layer1_neuron, layer2_neuron, layer3_neuron, layer4_neuron, lr, kernel_size, activation_function, dropout_value, opt_class)
        model.fit(train_images, train_labels, epochs=nb_epochs, validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        if (best_acc < test_acc):
            best_acc = test_acc
            best_lr = lr
            best_opt = opt_name
        print(f"Optimizer: {opt_name}, lr: {lr} - Test Accuracy: {test_acc*100:.2f}%\n")

print(f"\nBest combination: (Optimizer: {best_opt} - Learning Rate: {best_lr} - Accuracy: {best_acc*100:.4f})\n")