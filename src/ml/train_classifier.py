import tensorflow as tf
import keras
from keras import layers, models
import os

def build_and_train_model(train_texts, train_labels, num_classes, save_path):
    vectorize_layer = layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=200)
    vectorize_layer.adapt(train_texts)

    model = models.Sequential([
        vectorize_layer,
        layers.Embedding(10000, 64, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_texts, train_labels, epochs=5, batch_size=32, validation_split=0.2)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model