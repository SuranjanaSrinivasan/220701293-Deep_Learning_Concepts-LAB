import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test, history=None):
    """Evaluate a trained Keras model on test data and print basic metrics."""
    if hasattr(model, 'predict'):
        preds = model.predict(X_test)
        # if output is probability for binary
        if preds.shape[-1] == 1 or preds.ndim == 1:
            preds = (preds.ravel() >= 0.5).astype(int)
        else:
            preds = np.argmax(preds, axis=1)

    else:
        raise ValueError('model does not have predict method')

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_test, preds))

    return {'accuracy': acc, 'confusion_matrix': cm}
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, history):
    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Accuracy Plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()
