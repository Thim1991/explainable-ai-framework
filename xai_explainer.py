import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime import lime_tabular
import shap

class XAIExplainer:
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer_lime = None
        self.explainer_shap = None

    def train_lime_explainer(self, X_train):
        self.explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=\'classification\'
        )

    def explain_instance_lime(self, instance, num_features=5):
        if self.explainer_lime is None:
            raise ValueError("LIME explainer not trained. Call train_lime_explainer first.")
        explanation = self.explainer_lime.explain_instance(
            data_row=instance.values,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )
        return explanation.as_list()

    def train_shap_explainer(self, X_train):
        self.explainer_shap = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer_shap.shap_values(X_train)

    def explain_instance_shap(self, instance):
        if self.explainer_shap is None:
            raise ValueError("SHAP explainer not trained. Call train_shap_explainer first.")
        shap_values_instance = self.explainer_shap.shap_values(instance)
        return shap_values_instance

    def plot_shap_explanation(self, instance, instance_index=0):
        if self.explainer_shap is None or self.shap_values is None:
            raise ValueError("SHAP explainer not trained. Call train_shap_explainer first.")
        shap.initjs()
        return shap.force_plot(self.explainer_shap.expected_value[1], self.shap_values[1][instance_index,:], instance)

if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100) * 10,
        'feature_3': np.random.randint(0, 2, 100),
        'target': np.random.randint(0, 2, 100)
    })

    X = data[['feature_1', 'feature_2', 'feature_3']]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()
    class_names = ['Class 0', 'Class 1']

    explainer = XAIExplainer(model, feature_names, class_names)

    # LIME Explanation
    explainer.train_lime_explainer(X_train)
    instance_to_explain_lime = X_test.iloc[0]
    lime_explanation = explainer.explain_instance_lime(instance_to_explain_lime)
    print("LIME Explanation for instance:", lime_explanation)

    # SHAP Explanation
    explainer.train_shap_explainer(X_train)
    instance_to_explain_shap = X_test.iloc[0]
    shap_explanation = explainer.explain_instance_shap(instance_to_explain_shap)
    print("SHAP Explanation for instance (first class):")
    # Note: For actual plotting, you'd need to run this in an environment that can render plots
    # explainer.plot_shap_explanation(instance_to_explain_shap, instance_index=0)
