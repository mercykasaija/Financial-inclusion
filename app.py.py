from threading import local
import category_encoders as ce
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle

from xgboost import Booster

st.write('The objective of this app is to create a machine learning model to predict which individuals are most likely '
         'to have or use a bank account.')
st.write("The models and solutions developed can provide an indication of the state of financial inclusion in Kenya,"
         "Rwanda, Tanzania and Uganda,"
         "while providing insights into some of the key factors driving individuals’ financial security.")

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')
train_data = train_data.replace({'location_type': {0: "Rural", 1: "Urban"}})
train_data = train_data.replace({'cellphone_access': {0: "No", 1: "Yes"}})

test_data = test_data.replace({'location_type': {0: "Rural", 1: "Urban"}})
test_data = test_data.replace({'cellphone_access': {0: "No", 1: "Yes"}})

# train_data['age_bins'] = pd.cut(train_data['age_of_respondent'], bins=3,
#                               labels=['Young', 'Adult', 'Elderly'])
# test_data['age_bins'] = pd.cut(train_data['age_of_respondent'], bins=3,
#                              labels=['Young', 'Adult', 'Elderly'])
# Title
st.header('Financial Inclusion in Africa')
# Input data
st.date_input("Today's date")
gender_of_respondent = st.selectbox("What is your gender", ("Female", "Male"))
if gender_of_respondent == "Female":
    gender_of_respondent = 1
else:
    gender_of_respondent = 0
age_of_respondent = st.number_input("Please input your age", min_value=18, max_value=100)

year = st.selectbox("Select year", (2018, 2019, 2020))

country = st.selectbox("Select country of origin", ("Kenya", "Rwanda", "Tanzania", "Uganda"))
if country == "Kenya":
    country = 1
elif country == "Rwanda":
    country = 2
elif country == "Tanzania":
    country = 3
else:
    country = 4

# country.map({"Kenya":0,"Rwanda":1,"Tanzania":2,"Uganda":3}).astype(int)
location_type = st.selectbox("Do you live in the rural area or urban area", ("Rural", "Urban"))

if location_type == "Rural":
    location_type = 1
else:
    location_type = 0
cellphone_access = st.selectbox("Do you have access to cellphone", ("Yes", "No"))

if cellphone_access == "Yes":
    cellphone_access = 1
else:
    cellphone_access = 0
relationship_with_head = st.selectbox("What is your relationship with the head of the family", ("Head of Household",
                                                                                                "Spouse", "Child",
                                                                                                "Parent",
                                                                                                "Other relative",
                                                                                                "Other non_relative"))

if relationship_with_head == "Head of Household":
    relationship_with_head = 0
elif relationship_with_head == "Spouse":
    relationship_with_head = 1
elif relationship_with_head == "Child":
    relationship_with_head = 2
elif relationship_with_head == "Parent":
    relationship_with_head = 3
elif relationship_with_head == "Other relative":
    relationship_with_head = 4
else:
    relationship_with_head = 5

marital_status = st.selectbox("What is your marital status", ("Married/Living together", "Single/Never Married",
                                                              "Widowed", "Divorced/Separated", "Don't know"))

if marital_status == "Married/Living together":
    marital_status = 0
elif marital_status == "Single/Never Married":
    marital_status = 1
elif marital_status == "Widowed":
    marital_status = 2
elif marital_status == "Divorced/Separated":
    marital_status = 3
else:
    marital_status = 4

education_level = st.selectbox("Highest education-level", ("Primary education", "No formal education",
                                                           "Secondary education", "Tertiary education",
                                                           "Vocational/Specialised training", "Other/Don't know/RTA "
                                                           ))

if education_level == "Primary education":
    education_level = 0
elif education_level == "No formal education":
    education_level = 1
elif education_level == "Secondary education":
    education_level = 2
elif education_level == "Tertiary education":
    education_level = 3
elif education_level == "Vocational/Specialised training":
    education_level = 4
else:
    education_level = 5

job_type = st.selectbox("Indicate your job-type", ("Self employed", "Informally employed", "Farming and Fishing",
                                                   "Remittance Dependent", "Other Income", "Formally employed Private",
                                                   "No Income", "Formally employed Government", "Government Dependent",
                                                   "Don't Know/Refuse to answer"))

if job_type == "Self employed":
    job_type = 0
elif job_type == "Informally employed":
    job_type = 1
elif job_type == "Farming and Fishing":
    job_type = 2
elif job_type == "Remittance Dependent":
    job_type = 3
elif job_type == "Other Income":
    job_type = 4
elif job_type == "Formally employed Private":
    job_type = 5
elif job_type == "No Income":
    job_type = 6
elif job_type == "Formally employed Government":
    job_type = 7
elif job_type == "Government Dependent":
    job_type = 8
else:
    job_type = 9

household_size = st.number_input("what is your household-size", min_value=1, max_value=21)
# household_size.astype(int)

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(train_data)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_data['bank_account'], test_size=0.3,
                                                    random_state=0)

ord_enc = ce.OrdinalEncoder(cols=['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                                  'relationship_with_head', 'marital_status', 'education_level', 'job_type'
                                  ]).fit(X_train, y_train)

data = ord_enc.transform(all_data)
data.head()

main_cols = data.columns.difference(['bank_account'])

train = data[:ntrain].copy()
# train.drop_duplicates(inplace = True, ignore_index=True)
target = train.bank_account.copy()
train.drop(['bank_account', 'uniqueid'], axis=1, inplace=True)

test = data[ntrain:].copy()
test.drop(['bank_account', 'uniqueid'], axis=1, inplace=True)
test = test.reset_index(drop=True)
lr = LogisticRegression()
lr.fit(train, target)
preds = lr.predict(test)

# If button is pressed
if st.button("Submit"):

    X = pd.DataFrame(
        {"country": country, "location": location_type, "cellphone": cellphone_access, "household": household_size,
         "age": age_of_respondent, "gender": gender_of_respondent,
         "relationship": relationship_with_head, "marital": marital_status, "education": education_level,
         "job": job_type, "year": year}, index=[0]
    )
    # st.write("From the following data that you have given: ")
    # st.write(X)
    # X = X.replace(["No", "Yes"], [0, 1])
    # Get prediction
    prediction = lr.predict(X)[0]
    # predict_probability = lr.predict_proba(X)

    # Output prediction

    if prediction == 1:
        st.write(f'This respondent is likely to have a bank account.')
    else:
        st.write(
            f'This respondent is NOT likely to have a bank account.')
