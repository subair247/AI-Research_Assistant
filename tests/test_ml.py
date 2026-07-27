import pytest
from src.ml.predictor import DocumentClassifier

def test_classifier_fallback():
    classifier = DocumentClassifier(model_path="./models/non_existent_model.h5")
    prediction = classifier.predict("Artificial Intelligence and deep neural networks in healthcare.")
    
    assert prediction in [
        "Artificial Intelligence", "Machine Learning", "Computer Vision",
        "Natural Language Processing", "Robotics", "Cyber Security", "Cloud Computing"
    ]