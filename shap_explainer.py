import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_explain_model(X, y):
    """
    Trains a RandomForestClassifier and explains its predictions using SHAP.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.

    Returns:
        tuple: Trained model and SHAP explainer.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("SHAP values computed. You can visualize them using shap.summary_plot or shap.force_plot.")
    return model, explainer, shap_values, X_test

if __name__ == "__main__":
    # Example usage with dummy data (Iris dataset-like)
    data = {
        'sepal_length': np.random.rand(150) * 2 + 4,
        'sepal_width': np.random.rand(150) * 1.5 + 2,
        'petal_length': np.random.rand(150) * 3 + 1,
        'petal_width': np.random.rand(150) * 1 + 0.1,
        'species': np.random.randint(0, 3, 150)
    }
    df = pd.DataFrame(data)

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    model, explainer, shap_values, X_test = train_and_explain_model(X, y)

    # Example visualization (requires matplotlib, which is usually installed with shap)
    # shap.summary_plot(shap_values, X_test, plot_type="bar")
    # plt.savefig("shap_summary_plot.png")
    # plt.close()
    print("SHAP explainer for RandomForestClassifier implemented.")
