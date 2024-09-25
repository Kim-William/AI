import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
import numpy as np

# 1. Load the Cats and Dogs dataset
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
train_dataset = dataset['train']

# 2. Resize and normalize the images
def format_image(image, label):
    try:
        image = tf.image.resize(image, (150, 150))  # Resize the image
        image = image / 255.0  # Normalize
    except:
        return None, None
    return image, label

# Skip None images (corrupt images)
train_dataset = train_dataset.map(format_image)
train_dataset = train_dataset.filter(lambda x, y: x is not None)

train_size = int(0.8 * info.splits['train'].num_examples)
train_images = train_dataset.take(train_size)
validation_images = train_dataset.skip(train_size)

# Batch the data and prefetch for better performance
BATCH_SIZE = 32
train_images = train_images.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_images = validation_images.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 3. Define the VGG16-based model with Regularization, Dropout, and Early Stopping
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze VGG16 layers
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # L2 Regularization
        layers.Dropout(0.5),  # Dropout
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# 4. Initialize the model
model = build_model()

# 5. Set up Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 6. Train the model with the three techniques
model.fit(train_images, epochs=10, validation_data=validation_images, callbacks=[early_stopping])

# 7. Make predictions on the validation set
y_true = []
y_pred = []
y_prob = []

for images, labels in validation_images:
    predictions = model.predict(images)  # Get predicted probabilities
    y_true.extend(labels.numpy())  # Store true labels
    y_prob.extend(predictions)  # Store predicted probabilities
    
    # Convert probabilities to binary classification (0 or 1)
    pred_classes = [1 if prob > 0.5 else 0 for prob in predictions]
    y_pred.extend(pred_classes)

# 8. Evaluate the model
def evaluate_model(y_true, y_pred, y_prob):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision
    precision = precision_score(y_true, y_pred)
    
    # Recall
    recall = recall_score(y_true, y_pred)
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

# 9. Call the evaluation function
evaluate_model(y_true, y_pred, y_prob)
