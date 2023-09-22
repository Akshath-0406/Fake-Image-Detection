import tensorflow as tf

img_height = 32
img_width = 32
batch_size = 32

# Load the training data
print("Loading the data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  "data/train",
  seed = 63,
  image_size = (img_height, img_width),
  batch_size = batch_size)

# Load the validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
  "data/test",
  seed = 63,
  image_size = (img_height, img_width),
  batch_size = batch_size)

# Quick check to make sure it's all loaded properly
print("Training Classes:")
class_names = train_ds.class_names
print(class_names)

print("Testing Classes:")
class_names = train_ds.class_names
print(class_names)

# Building the CNN
layers = []
layers.append(tf.keras.layers.Rescaling(1./255)) # Normalise pixel values
layers.append(tf.keras.layers.Conv2D(32, 3, activation='relu'))
layers.append(tf.keras.layers.MaxPooling2D())
layers.append(tf.keras.layers.Flatten())

# Building the ANN
layers.append(tf.keras.layers.Dense(64, activation='relu'))
layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

# Create and compile the model from layers
model = tf.keras.Sequential(layers)
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Build the model so we can see a summary
model.build(input_shape=(None, 32, 32, 3))
model.summary()


# Training the model
print("Starting training...")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5,
  verbose=1
)
print("Training finished.")
model.save("model.h5")