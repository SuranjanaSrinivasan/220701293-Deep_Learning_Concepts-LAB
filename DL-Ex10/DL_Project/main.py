from src.train_model import train_model
from src.evaluate_model import evaluate_model

if __name__ == "__main__":
    model, X_test, y_test, history = train_model()
    evaluate_model(model, X_test, y_test, history)
