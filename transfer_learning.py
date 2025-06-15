import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = 'data/star-wars/'
names = ['YODA', 'LUKE SKYWALKER', 'R2-D2', 'MACE WINDU', 'GENERAL GRIEVOUS']

# Use VGG19 without the classification head
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
base_model.trainable = False

# Build the model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5)  # No softmax here; handled by loss function
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# ImageDataGenerator setup
preprocess_input = tf.keras.applications.vgg19.preprocess_input

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_batches = train_gen.flow_from_directory(
    BASE_DIR + 'train/',
    target_size=(256, 256),
    batch_size=4,
    class_mode='sparse',
    shuffle=True,
    color_mode='rgb',
    classes=names,
)
valid_batches = valid_gen.flow_from_directory(
    BASE_DIR + 'val/',
    target_size=(256, 256),
    batch_size=4,
    class_mode='sparse',
    shuffle=False,
    color_mode='rgb',
    classes=names,
)
test_batches = test_gen.flow_from_directory(
    BASE_DIR + 'test/',
    target_size=(256, 256),
    batch_size=4,
    class_mode='sparse',
    shuffle=False,
    color_mode='rgb',
    classes=names,
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)

# Train the model
history = model.fit(train_batches, epochs=30, callbacks=[early_stopping], validation_data=valid_batches, verbose=2)

# Evaluate
model.evaluate(test_batches, verbose=2)

# Predict
predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

# Display
def show(batch, pred_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(batch[0][i].astype('uint8'))  # Optional: un-preprocess
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl += '/Pred:' + names[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()

print(test_batches[0][1])
print(labels[0:4])

show(test_batches[0], labels[0:4])
