import warnings
warnings.filterwarnings("ignore")
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.losses import binary_crossentropy
from keras.applications import ResNet50
import matplotlib.pyplot as plt

# Define paths for data
train_path = r'tcga_coad_msi_mss/train'
test_path = r'tcga_coad_msi_mss/test'
val_path = r'tcga_coad_msi_mss/val'

# Image size and batch size
image_size = (150, 150)
batch_size = 32

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Data augmentation for test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Data augmentation for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Labels
labels = {value: key for key, value in train_generator.class_indices.items()}

print("Labels in train and validation datasets\n")
for key, value in labels.items():
    print(f'{key} : {value}')

# Transfer learning with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Training the model
with tf.device('/GPU:0'):
    start = timer()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=[reduce_lr, early_stopping, model_checkpoint],
        verbose=1
    )
    end = timer()

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model
result_1 = model.evaluate(test_generator)
print(f'Test Loss: {result_1[0]}')
print(f'Test Accuracy: {result_1[1]}')

# Plot training and validation loss
plt.figure(figsize=[6, 4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Training_And_Validation_Loss')
plt.savefig('Training_And_Validation_Loss_ResNet50.png')

# Make predictions
test_predictions = model.predict(validation_generator)
test_predictions_classes = np.where(test_predictions > 0.5, 1, 0)

# Print classification report
from sklearn.metrics import classification_report
y_true = validation_generator.classes
print(classification_report(y_true, test_predictions_classes, target_names=labels.values()))

