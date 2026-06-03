import tensorflow as tf
from tensorflow.keras import layers, models

# 1. SETUP THE DATASET
dataset_path = "./dataset"
batch_size = 16
img_size = (224, 224)

print("Loading your custom images...")
# Load the training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Load the validation data (used to test the AI during training)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Get the names of your folders (your custom categories)
class_names = train_dataset.class_names
print(f"Categories found: {class_names}")

# 2. BUILD THE CUSTOM AI BRAIN
print("Building Custom MobileNetV2 Model...")
# Load the base model, but chop off the top layer (include_top=False)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model so we don't destroy its pre-trained visual knowledge
base_model.trainable = False

# Build our new custom top layers
model = models.Sequential([
    # Step A: Data Augmentation (slight rotations/zooms to prevent memorization)
    layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    # Step B: Preprocessing math for MobileNetV2
    layers.Rescaling(1./127.5, offset=-1),
    
    # Step C: The frozen base model
    base_model,
    
    # Step D: Our new custom classification layers
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Drops 20% of connections to force the AI to learn harder
    layers.Dense(len(class_names), activation='softmax') # The final output layer
])

# 3. TRAIN THE AI
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- STARTING TRAINING ---")
# Epochs = How many times the AI studies the whole dataset. 10 is good for beginners.
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# 4. SAVE YOUR MASTERPIECE
model.save("my_custom_brain.keras")
print("\nTraining Complete! Saved as 'my_custom_brain.keras'")

# Save the category names so your API knows what they are
with open("class_names.txt", "w") as f:
    f.write(",".join(class_names))