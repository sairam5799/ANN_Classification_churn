import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

#load the trained model
model=tf.keras.models.load_model('model.h5')

##load encoder and scalar
with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scalar.pkl','rb') as file:
   scalar=pickle.load(file)

   ##streamlit

st.title('Customer Churn Prediction')

##user input
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balence=st.number_input('Balence')
credit_score=st.number_input('Credit score')
estimated_salary=st.number_input('Estimated Salary')
tenure= st.slider('Tenure',0, 10)
num_of_products=st.slider=('Number of Products',1,4)
has_cr_card=st.selectbox('Has credit Card', [0, 1])
is_active_member= st.selectbox('Is Active Member', [0, 1])

##prepare input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balence': [balence],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
    })
#One Hot Encoding Geography
geo_encoded=label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

##combine One hot Encoder with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

##scale the input data
input_data_scaled=scalar.transform(input_data)

##predict churn
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

if prediction_prob > 0.5:
    st.write("the customer is likely to churn")
else:
    st.write("the customer is not likely to churn")
