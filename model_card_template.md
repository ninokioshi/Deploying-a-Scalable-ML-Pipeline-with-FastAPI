# Model Card: Census Income Prediction API

## Model Details
- **Developer:** Student (ninokioshi)
- **Model Type:** Supervised binary classification
- **Algorithm:** Random Forest Classifier (from scikit-learn)
- **Version:** 1.0
- **Training Script:** `train_model.py`
- **Primary Dependencies:** scikit-learn, pandas, numpy, pytest, fastapi, uvicorn
- **Repository:** [GitHub - ninokioshi/Deploying-a-Scalable-ML-Pipeline-with-FastAPI](https://github.com/ninokioshi/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

---

## Intended Use
This model predicts whether an individual earns **more than $50K** or **less than or equal to $50K** per year based on demographic and employment-related attributes from the **U.S. Census Adult Income Dataset**.

- **Intended Users:** Data scientists, analysts, and API consumers interested in automated income classification.
- **Intended Use Cases:** 
  - Demonstrating scalable model deployment using FastAPI and CI/CD.
  - Educational and academic use only.

---

## Data
- **Dataset:** [Census Income Dataset (Adult Data Set from UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/adult)
- **Size:** ~32,000 records used for training and testing.
- **Features Used:**
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
  - Continuous: Remaining numerical features
- **Target:** `salary` — binary labels (`<=50K` or `>50K`)

---

## Model Performance
| Metric     | Value  |
|-------------|---------|
| Precision   | 0.7338 |
| Recall      | 0.6365 |
| F1 Score    | 0.6817 |

These metrics were computed on the test dataset using the model trained via `train_model.py`.

Additionally, model slice performance (saved in `slice_output.txt`) was evaluated across categorical feature subgroups to check for consistency and fairness.

---

## Ethical Considerations
The dataset contains sensitive features such as **race**, **sex**, and **native-country**, which may reflect societal biases.  
As a result, model predictions could unintentionally reflect or amplify existing inequalities.  

Users should:
- Avoid using the model for real-world decision-making without additional fairness and bias analysis.
- Consider removing sensitive attributes for production use.

---

## Caveats and Recommendations
- Model performance is limited to the 1994 U.S. Census dataset and may not generalize to modern populations.
- Predictions should be interpreted with caution due to potential bias in the dataset.
- For real deployment, implement:
  - Model retraining with updated data
  - Feature scaling and hyperparameter tuning
  - Continuous monitoring via API endpoints

---

## Acknowledgements
This project is part of **Udacity’s Machine Learning DevOps Engineer Nanodegree (Project D501-2)**.  
It demonstrates end-to-end deployment of a scalable ML pipeline using **FastAPI**, **GitHub Actions**, and **pytest**.
