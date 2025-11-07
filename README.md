# Deploying a Scalable ML Pipeline with FastAPI

## Author
**Nino Delgado**

## Project Overview
This project trains and deploys a machine learning model to predict whether an individual earns **more than $50K** or **less than or equal to $50K** annually, based on demographic and employment data from the U.S. Census.  
It demonstrates the full ML lifecycle — data preprocessing, model training, evaluation, testing, and API deployment — while maintaining CI/CD automation through GitHub Actions.

## Repository
**Public GitHub Repository:**  
[https://github.com/ninokioshi/Deploying-a-Scalable-ML-Pipeline-with-FastAPI](https://github.com/ninokioshi/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

---

## Environment Setup
1. Create and activate a virtual environment:
   python3 -m venv fastapi  
   source fastapi/bin/activate  
   pip install -r requirements.txt  

---

## Training and Evaluation
Run the following command to train the model and compute evaluation metrics:  
   python train_model.py  

**Outputs:**
- model/model.pkl — trained Random Forest model  
- model/encoder.pkl — fitted OneHotEncoder  
- slice_output.txt — metrics across categorical slices  
- Console output showing model performance  

**My Results:**  
Precision: 0.7338 | Recall: 0.6365 | F1: 0.6817  

---

## Unit Testing
Run all tests:  
   pytest -v  

**Expected result:**  
3 passed in 2.71s  

**Screenshot saved:**  
screenshots/unit_test.png  

---

## Continuous Integration (CI)
This project uses GitHub Actions for continuous integration.  
Every push to the main branch triggers the following steps:  
- Install dependencies from requirements.txt  
- Run flake8 for linting  
- Run pytest for testing  

**Screenshot saved:**  
screenshots/continuous_integration.png  

---

## FastAPI Deployment
Start the API locally:  
   uvicorn main:app --reload  

Then, in another terminal, run the local API client:  
   python local_api.py  

**Example responses:**  
GET Status Code: 200  
GET Response: {'message': 'Welcome to the Census Income Prediction API!'}  
POST Status Code: 200  
POST Response: {'prediction': '>50K'}  

**Screenshot saved:**  
screenshots/local_api.png  

---

## Model Card
A complete model card is located in:  
model/model_card_template.md  

It includes:  
- Model overview and purpose  
- Dataset details  
- Evaluation metrics and slice analysis  
- Ethical considerations and intended use  

---

## Data Slice Evaluation
Model performance across data slices is written to slice_output.txt.  
Each slice represents one unique categorical feature value, showing how the model performs for subsets of the data.  

**Example:**  
workclass: Private, Count: 4595  
Precision: 0.7381 | Recall: 0.6245 | F1: 0.6766  

---

## Project Deliverables Checklist
✅ ml/data.py — Preprocessing functions implemented  
✅ ml/model.py — Model training, inference, and slice performance functions implemented  
✅ train_model.py — End-to-end ML pipeline completed  
✅ test_ml.py — 3+ passing unit tests  
✅ main.py and local_api.py — API endpoints implemented and tested locally  
✅ model_card_template.md — Completed with metrics and documentation  
✅ screenshots/ — Includes:  
   - continuous_integration.png  
   - unit_test.png  
   - local_api.png  

---

## Submission Instructions
Before submission, ensure:  
- Your GitHub repository is public  
- The README includes your GitHub link  
- The latest commit contains:  
  - All code and screenshots  
  - Model card and slice metrics  
  - Passing CI build  

**Final Deliverable:**  
A fully functional, tested, and documented machine learning pipeline deployed with FastAPI, demonstrating CI/CD integration and data slice analysis.
