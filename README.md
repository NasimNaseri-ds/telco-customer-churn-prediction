Project overview:
This project analyzes customer behavior and predicts customer churn (i.e., whether a customer will leave the telecom company) using the Telco Customer Churn Dataset.
The main objective is to identify the key factors influencing churn and to build a reliable machine learning model that helps the company improve customer retention strategies.
The dataset contains 7,043 rows and 21 columns, each representing various customer attributes.
________________________________________
Dataset:
Source: Telco Customer Churn Dataset on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Each record in the dataset represents a single customer, including features such as:
•	Customer demographics (e.g., gender, partner, dependents)
•	Account details (e.g., tenure, payment method, monthly charges)
•	Services subscribed (e.g., Internet, streaming, security)
•	Churn label indicating whether the customer left the company
________________________________________
 Data Cleaning and Preprocessing:
The preprocessing phase involved:
1.	Data inspection: Checked dataset shape, data types, and duplicate rows.
2.	Column standardization: Converted all column names to lowercase for consistency.
3.	Check for missing values and repeated information:
o	The totalcharges deleted because it was the production of tenure and monthlycharges (in feature engineering part choosing just totalcharges or having both tenure and monthlycharges is considered) .
o	The columns (onlinesecurity, onlinebackup, deviceprotection,techsupport) had repeated informations, so I combined them and then OneHot code the new column.  
________________________________________
 Model Selection:
For this project, a CatBoostClassifier was chosen due to its efficiency in handling categorical variables and strong performance on tabular data without requiring extensive preprocessing.
The model was trained using the training and validation sets and a confusion matrix was plotted to visualize the accurecies.________________________________________
Feature engineering:
Although the CatBoost model achieved 80% accuracy, the recall for churned customers (TP rate) indicated room for improvement.  Optimization steps include:
1)	Balancing the target column with class wights,
2)	Feature importance analysis to better understand churn customers
•	Created a new composite feature family_status from seniorcitizen, partner, and dependents, which simplified related variables without loss of information.
•	Performed correlation analysis and permutation testing to identify low-impact features such as gender, streamingtv, and multiplelines, which were later removed.
________________________________________
Conclusion:
Through feature engineering and hyperparameter tuning using RandomizedSearchCV, the F1-score for detecting churned customers improved significantly—from 55% in the initial model to 62% in the final model. As expected, overall accuracy decreased from 80% to 74% due to the application of class weights, prioritizing the identification of churned customers. 

