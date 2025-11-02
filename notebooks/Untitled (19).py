
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib


df_path = '/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(df_path)
df.head()

print(df.shape)


#DataFrame Cleaning


num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")
df['customerID'].duplicated().any()
df.columns
df.columns = df.columns.str.lower()
print(df.dtypes)


#features cleaning


df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
for col in df.select_dtypes(include=['float', 'int']).columns:
    missing_count = df[col].isnull().sum()
    print(f"Column '{col}' has {missing_count} missing values")

sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
sns.histplot(df['totalcharges'], bins=30, kde=True, color='lightgreen')

plt.title("Distribution of totalcharges", fontsize=16)
plt.xlabel("totalcharges", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.show()

correlation = df['totalcharges'].corr(df['monthlycharges'])

print(f"Correlation between totalcharges and monthlycharges: {correlation:.2f}")

plt.figure(figsize=(10,6))
sns.scatterplot(x='monthlycharges', y='totalcharges', data=df, alpha=0.5)
plt.title("totalcharges vs monthlycharges", fontsize=16)
plt.xlabel("monthlycharges", fontsize=12)
plt.ylabel("totalcharges", fontsize=12)
plt.show()
monthly_times_tenure = df['monthlycharges'] * df['tenure']

correlation = df['totalcharges'].corr(monthly_times_tenure)
print(f"Correlation between totalcharges and monthlycharges × Tenure: {correlation:.2f}")
#totalcharges gives us no new information, instead of filling the NaN values in this column, we deled this column.
df = df.drop('totalcharges', axis=1)


for col in df.select_dtypes(include='object').columns:
    missing_count = df[col].isnull().sum() + (df[col] == '').sum()
    print(f"Column '{col}' has {missing_count} missing/empty values")

df.head()

#feature design


df['multiplelines'] = df['multiplelines'].replace({
    'No phone service': 'No service',
    'No': 'Single line',
    'Yes': 'Multiple lines'
})


print(df['multiplelines'].value_counts())

df = df.drop(columns=['phoneservice'])


internet_related = ['onlinesecurity', 'onlinebackup',
                    'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']

def combine_internet_services(row):
    if row['internetservice'].lower() == 'no':
        return ['No internet']
    else:
        services = []
        for col in internet_related:
            if str(row[col]).lower() == 'yes':
                services.append(col)
        return services if services else ['No additional services']

df['internetfeatures'] = df.apply(combine_internet_services, axis=1)


print(df['internetfeatures'].value_counts())

columns_to_drop = ['onlinesecurity', 'onlinebackup', 
                   'deviceprotection', 'techsupport', 
                   'streamingtv', 'streamingmovies']  

df = df.drop(columns=columns_to_drop)

df.head()

print(df.shape)

df['internetfeatures'] = df['internetfeatures'].apply(lambda x: x if isinstance(x, list) else [x])

mlb = MultiLabelBinarizer()

internet_onehot = pd.DataFrame(mlb.fit_transform(df['internetfeatures']),
                               columns=mlb.classes_,
                               index=df.index)

df = df.join(internet_onehot)

df.head()


#No additional services is givving us repeted information, so it will be deleted.

df=df.drop(columns= ['No additional services','internetfeatures'])


df.columns


df_features = df.drop(columns=['customerid', 'churn']) 
churn_target = df['churn']


X_train_full_v1, X_test_v1, y_train_full_v1, y_test_v1 = train_test_split(
    df_features, churn_target, 
    test_size=0.2, random_state=42, stratify=churn_target
)

X_train_v1, X_val_v1, y_train_v1, y_val_v1 = train_test_split(
    X_train_full_v1, y_train_full_v1, 
    test_size=0.25, random_state=42, stratify=y_train_full_v1
)


cat_features_v1 = df_features.select_dtypes(include=['object', 'category']).columns.tolist()



cat_model_v1 = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,                
    eval_metric='Accuracy',  
    random_state=42,
    verbose=100              
)

cat_model_v1.fit(X_train_v1, y_train_v1, cat_features=cat_features_v1, eval_set=(X_val_v1, y_val_v1))


y_pred_v1 = cat_model_v1.predict(X_test_v1)


print("\nClassification Report:\n")
print(classification_report(y_test_v1, y_pred_v1))


cm = confusion_matrix(y_test_v1, y_pred_v1, labels=['No','Yes'])


plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not churn','Churn'], yticklabels=['Not churn','Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#The most important of this matrix is the TP (yes,recall in confution matrix), becouse our goal is to predict churn cutomer. So our model is not good enugh.

#Feature engineering.


count_churn = df['churn'].value_counts()
print(count_churn)

#this column is not balanced, so the f1 score is low.

le = LabelEncoder()
churn_target_encoded = le.fit_transform(churn_target)

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
    df_features, churn_target_encoded, 
    test_size=0.2, random_state=42, stratify=churn_target_encoded
)

cat_features_v2 = df_features.select_dtypes(include=['object', 'category']).columns.tolist()

total = len(df)
no_count = 5174
yes_count = 1869
n_classes = 2

weight_no = total / (n_classes * no_count)
weight_yes = total / (n_classes * yes_count)

print(weight_no, weight_yes)

scoring='f1'


# In[58]:


cat = CatBoostClassifier(
    class_weights=[weight_no, weight_yes],
    eval_metric='F1',
    verbose=0,
    random_state=42
)

params = {
    'depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [500, 800, 1000]
}


cat_model_v2 = RandomizedSearchCV(
    cat,
    param_distributions=params,
    n_iter=15,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42
)

cat_model_v2.fit(
    X_train_v2, 
    y_train_v2, 
    cat_features=cat_features_v2
)


print("Best parameters:", cat_model_v2.best_params_)
print("Best F1 score:", cat_model_v2.best_score_)



y_pred_v2 = cat_model_v2.predict(X_test_v2)


print("\nClassification Report:\n")
print(classification_report(y_test_v2, y_pred_v2))


#good improvement in f1 score for yes(recognising churn customers).

#we need corelations to do more feature engineering.

df_encode = df_features.copy()

binary_cols=['paperlessbilling' , 'partner' , 'dependents']


for col in binary_cols:
    df_encode[col] = df_encode[col].str.strip().str.capitalize()
for col in binary_cols:
    df_encode[col] = df_encode[col].map({'Yes': 1, 'No': 0})

df_encode['gender'] = df_encode['gender'].map({'Male': 1, 'Female': 0})

df_encode = pd.get_dummies(df_encode, columns=['paymentmethod'], prefix='payment', dtype=int)

df_encode = pd.get_dummies(df_encode, columns=['multiplelines'], prefix='multiplelines', dtype=int)

df_encode = pd.get_dummies(df_encode, columns=['internetservice'], prefix='internet', dtype=int)

df_encode = pd.get_dummies(df_encode, columns=['contract'], prefix='contract', dtype=int)

df_encode = df_encode.drop(['internet_No'], axis=1)


df_encode.head()

churn_target_series=pd.Series(churn_target_encoded)

df_encode.corrwith(churn_target_series)


for col in [ 'partner', 'dependents']:
    df_features[col] = df_features[col].map({'No':0, 'Yes':1})


df_features['family_status'] = -df_features['seniorcitizen'] + df_features['partner'] + df_features['dependents']


df_features.head()


df_features['family_status'].corr(pd.Series(churn_target_encoded))


df_features = df_features.drop(columns=['partner', 'dependents','seniorcitizen'])


X_train_v3, X_test_v3, y_train_v3, y_test_v3 = train_test_split(
    df_features, churn_target_encoded, 
    test_size=0.2, random_state=42, stratify=churn_target_encoded
)


cat_features_v3 = df_features.select_dtypes(include=['object', 'category']).columns.tolist()

cat_model_v3 = RandomizedSearchCV(
    cat,
    param_distributions=params,
    n_iter=15,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42
)


cat_model_v3.fit(
    X_train_v3, 
    y_train_v3, 
    cat_features=cat_features_v3
)


print("Best parameters:", cat_model_v3.best_params_)
print("Best F1 score:", cat_model_v3.best_score_)

y_pred_v3 = cat_model_v3.predict(X_test_v3)


print("\nClassification Report:\n")
print(classification_report(y_test_v3, y_pred_v3))

#family_status could reduce dimensionality without hurting the model. So we keep this.


df_features['total_charges'] = df_features['tenure'] * df_features['monthlycharges']


df_features.head()


df_features['total_charges'].corr(pd.Series(churn_target_encoded))


df_features['monthlycharges'].corr(pd.Series(churn_target_encoded))


df_features['tenure'].corr(pd.Series(churn_target_encoded))

#total_charges has less correlaton than monthlycharges and tenure, so we will drop it.

df_features = df_features.drop(columns=['total_charges'])

#we go to test the permutation of ['streamingmovies','streamingtv','multiplelines','gender'] one by one.


X_test_shuffled_gender = X_test_v3.copy()
X_test_shuffled_gender['gender'] = np.random.permutation(X_test_shuffled_gender['gender'])


y_pred_shuffled_gender = cat_model_v3.predict(X_test_shuffled_gender)

print("\nClassification Report:\n")
print(classification_report(y_test_v3, y_pred_shuffled_gender))

print("\nClassification Report:\n")
print(classification_report(y_test_v3, y_pred_v3))


#This tells us that gender has almost no impact on our model’s predictions.We will drop it.

X_test_shuffled_streamingmovies = X_test_v3.copy()
X_test_shuffled_streamingmovies['streamingmovies'] = np.random.permutation(X_test_shuffled_streamingmovies['streamingmovies'])


y_pred_shuffled_streamingmovies = cat_model_v3.predict(X_test_shuffled_streamingmovies)


print("\nClassification Report:\n")
print(classification_report(y_test_v3, y_pred_shuffled_streamingmovies))

#shuffling streamingtv caused a small increase in accuracy and yes f1, we will not drop it.


X_test_shuffled_streamingtv = X_test_v3.copy()
X_test_shuffled_streamingtv['streamingtv'] = np.random.permutation(X_test_shuffled_streamingtv['streamingtv'])


y_pred_shuffled_streamingtv = cat_model_v3.predict(X_test_shuffled_streamingtv)

print("\nClassification Report:\n")
print(classification_report(y_test_v3, y_pred_shuffled_streamingtv))

#This tells us that streamingtv has almost no impact on our model’s predictions.We will drop it.


X_test_shuffled_multiplelines = X_test_v3.copy()
X_test_shuffled_multiplelines['multiplelines'] = np.random.permutation(X_test_shuffled_multiplelines['multiplelines'])


y_pred_shuffled_multiplelines = cat_model_v3.predict(X_test_shuffled_multiplelines)


print("\nClassification Report:\n")
print(classification_report(y_test_v3, y_pred_shuffled_multiplelines))


#This tells us that multiplelines has almost no impact on our model’s predictions.We will drop it.


df_features = df_features.drop(columns=['gender' , 'streamingtv' , 'multiplelines'])


X_train_v4, X_test_v4, y_train_v4, y_test_v4 = train_test_split(
    df_features, churn_target_encoded, 
    test_size=0.2, random_state=42, stratify=churn_target_encoded
)


cat_features_v4 = df_features.select_dtypes(include=['object', 'category']).columns.tolist()


cat_model_v4 = RandomizedSearchCV(
    cat,
    param_distributions=params,
    n_iter=15,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42
)


cat_model_v4.fit(
    X_train_v4, 
    y_train_v4, 
    cat_features=cat_features_v4
)



print("Best parameters:", cat_model_v4.best_params_)
print("Best F1 score:", cat_model_v4.best_score_)


y_pred_v4 = cat_model_v4.predict(X_test_v4)


print("\nClassification Report:\n")
print(classification_report(y_test_v4, y_pred_v4))

