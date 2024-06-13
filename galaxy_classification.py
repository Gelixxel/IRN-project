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

## If you want to test all of the different combinations of parameters you need to add for loops that iterates
## on each lists of parameters that are lines : 48, 55, 57 and 59. You alos can just change the values in lines 44 to 46 and 50 to 53

# Parameters
nb_epochs = 15 #15 # Change the number of epochs to vary the speed of progression

kernel_size = (3, 3) #(3, 3)
activation_function = 'relu' #'relu'
dropout_value = 0.5 #0.5

neuron_layers = [[8, 16, 32, 16], [16, 32, 64, 32], [32, 64, 128, 64], [64, 128, 256, 128]] # Add others neuron layers sizes to test

layer1_neuron = neuron_layers[2][0]
layer2_neuron = neuron_layers[2][1]
layer3_neuron = neuron_layers[2][2]
layer4_neuron = neuron_layers[2][3]

# kernel_sizes = [(3, 3), (5, 5), (7, 7)] # Add others kernel sizes to test

# dropout_values = [0.25, 0.5, 0.75] # Add others dropout values to test

# activation_functions = ['relu', 'sigmoid'] # Add others activation functions to test

optimizers = {
    # 'SGD': tf.keras.optimizers.SGD,
    # 'Adagrad': tf.keras.optimizers.Adagrad,
    'Adam': tf.keras.optimizers.Adam
    # Add others optimizers to test
}

learning_rates = [1e-3]#, 1e-2, 1e-1] # Add others learning rates to test

print("\nLoading Dataset\n")
dataset = load_galaxy_data()
train_images, test_images, train_labels, test_labels = preprocess_data(dataset)

best_opt = ''
best_lr = 0
best_kernel_size = (0, 0)
best_dropout_value = 0
best_activation_function = ''
best_neuron_layer_combination = []
best_acc = 0

print("\nIterate on each optimizer and each learning_rate\n")
for opt_name, opt_class in optimizers.items():
    for lr in learning_rates:
        # Create a logs directory
        log_dir = f"runs/{opt_name}_lr_{lr}_ksize_{kernel_size}_dropout_{dropout_value}_activation_{activation_function}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
            best_kernel_size = kernel_size
            best_dropout_value = dropout_value
            best_activation_function = activation_function
            best_neuron_layer_combination = neuron_layers[2]
        print(f"Optimizer: {opt_name} | lr: {lr} | Kernel size: {kernel_size} | Dropout value: {dropout_value} | Activation function: {activation_function} | Neuron layer combination: {neuron_layers[2]}\nTest Accuracy: {test_acc*100:.2f}%\n")

print(f"\nBest combination: (Accuracy: {best_acc*100:.4f}% - Optimizer: {best_opt} - Learning Rate: {best_lr} - Kernel size: {best_kernel_size} - Dropout value: {best_dropout_value} - Activation function: {best_activation_function} - Neuron layer combination: {best_neuron_layer_combination})\n")