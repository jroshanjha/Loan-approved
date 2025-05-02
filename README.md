# Employee-loan-deployment
This is ML Loan Flask Development for apply employees loan is approved or not.

Objective: Predict employees' loan eligibility based on profile data such as income, credit score, and employment history.

• Data Preparation & Preprocessing: Cleaned and preprocessed raw data, handling missing values, encoding categorical features, and normalizing numerical fields. Performed feature selection to retain the most relevant attributes for model performance.

• Imbalanced Data Handling: Detected class imbalance in the target variable (eligible vs ineligible).
Applied SMOTE (Synthetic Minority Oversampling Technique) and undersampling methods to balance the dataset.

• Model Comparison: Trained and compared multiple ML algorithms Logistic Regression, Random Forest, XGBoost, KNN, and SVM after evaluated using metrics: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

• Hyperparameter Tuning: Applied GridSearchCV and RandomizedSearchCV for fine-tuning model parameters and thn used 5-fold cross-validation to ensure generalization and reduce overfitting.

• Performance Boost: After applying imbalance correction and hyperparameter tuning, model performance improved by 10% in terms of F1-score and AUC.

• Final Outcome: XGBoost delivered the best results and was selected as the final model for deployment.

• Using CI/CD Pipeline for delplyment using Flask ( Docker with Github repository )

# Git Clone....
git clone https://github.com/ 

git config --global user.name "jroshanjha"
git config --global user.email "jroshan731@gmail.com"

## Create Virtual Environments & activate:- 
python -m venv code
code/Scripts/activate

### Emplyees Loan Amount Prediction Database:- http://www.kagle.com/datasets/

# Install dependencies
pip install -r requirements.txt


## Targets variables:-  loan_status 

## Independent Variables:-
person_age	person_gender	person_education	person_income	person_emp_exp	person_home_ownership	loan_amnt	loan_intent

brew install git-lfs              # or download from https://git-lfs.github.com/
git lfs install
git lfs track "models/trained_model.pkl"  -- ".pkl"
git add .gitattributes
git add models/trained_model.pkl 
git commit -m "Add trained model with Git LFS"
git push origin development

git lfs push --all origin main

