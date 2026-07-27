import os
import tensorflow as tf
import numpy as np

class DocumentClassifier:
    def __init__(self, model_path: str):
        self.categories = [
            "Artificial Intelligence", "Machine Learning", "Computer Vision",
            "Natural Language Processing", "Robotics", "Cyber Security", "Cloud Computing"
        ]
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None

    def predict(self, text_sample: str) -> str:
        if not self.model or not text_sample.strip():
            return "Artificial Intelligence"  
        
        preds = self.model.predict(np.array([text_sample]))
        class_idx = np.argmax(preds[0])
        return self.categories[class_idx % len(self.categories)]