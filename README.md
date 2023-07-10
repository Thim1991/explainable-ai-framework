# Explainable AI Framework

This framework provides tools and methodologies for interpreting and explaining the predictions of complex machine learning models. It integrates popular XAI techniques like LIME and SHAP to provide insights into model behavior.

## Features

- **LIME Integration**: Local Interpretable Model-agnostic Explanations for understanding individual predictions.
- **SHAP Integration**: SHapley Additive exPlanations for global and local interpretability.
- **Model Agnostic**: Can be used with any machine learning model.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### LIME Explanation Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xai_explainer import XAIExplainer
import numpy as np

# Dummy data
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
explainer.train_lime_explainer(X_train)

instance_to_explain = X_test.iloc[0]
lime_explanation = explainer.explain_instance_lime(instance_to_explain)
print("LIME Explanation:", lime_explanation)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
