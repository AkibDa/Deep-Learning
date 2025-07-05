from tensorflow.keras import layers, models

#A feedforward neural network (FNN) is the simplest form of neural network.
#             In these networks, information flows in one direction—from the input layer,
#                           through the hidden layers, to the output layer—without any feedback loops.



# Simple feedforward neural network
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

#Convolutional neural networks (CNNs) are specialized for processing grid-like data such as images.
#             CNNs use convolutional layers to detect patterns automatically in data, such as edges, textures, and shapes.

# Convolutional Neural Network
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Recurrent neural networks (RNNs) are designed for sequential data, such as time series or language.
#             Unlike FNNs, RNNs maintain a "memory" of previous inputs by passing the output of one layer back into the network.

# Simple RNN
model = models.Sequential([
    layers.SimpleRNN(128, input_shape=(100, 1)),
    layers.Dense(10, activation='softmax')
])

#Generative adversarial networks (GANs) consist of two networks, a generator and a discriminator, that are trained simultaneously.
#             The generator creates fake data, and the discriminator attempts to distinguish between real and generated data.
#                           Over time, the generator improves its ability to produce realistic data.

# GAN architecture (simplified)
generator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(100,)),
    layers.Dense(784, activation='sigmoid')
])

discriminator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(1, activation='sigmoid')
])

#Autoencoders are unsupervised learning models used for data compression.
#           They consist of an encoder that compresses the input data into a lower-dimensional representation and a decoder that reconstructs the original data from this representation.

# Simple Autoencoder
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

autoencoder = models.Model(input_img, decoded)