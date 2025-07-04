import tensorflow as tf
from tensorflow.keras import layers, models

# Load the Fashion MNIST dataset
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#Normalizing the image data ensures that the neural network trains efficiently, as the pixel values range from 0 to 1 instead of from 0 to 255.

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer to flatten the 2D images
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(64, activation='relu'),  # Additional hidden layer with 64 neurons
    layers.Dense(32, activation='relu'),   # Additional hidden layers with 32 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 classes
])

print(model.summary())

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Since this is a classification task, we will use sparse categorical crossentropy.
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_activityels, epochs=10, batch_size=32, verbose=2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_activityels)

print(f'Test accuracy: {test_acc}')