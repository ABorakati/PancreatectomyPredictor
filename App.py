# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:20:38 2020

@author: Dr Aditya Borakati
"""
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model('xgb')


def predict(model, input_df):
    predict_df = predict_model(estimator=model, data=input_df)
    predictions = predict_df['Score'][0]
 ###   percentrisk = print("%.2f%%" % (predictions * 100.0))
    return predictions

st.set_option('deprecation.showfileUploaderEncoding', False)



def run():


    add_selectbox = st.sidebar.selectbox(
    "Who are you predicting for?",
    ("An Individual", "Multiple Patients"))

    st.sidebar.info('This app has been created to predict the risk of developing diabetes mellitus in one year after having a partial pancreatectomy. A partial pancreatectomy is a surgical procedure to remove part of your pancreas, a leaf shaped organ in your abdomen which is important in blood sugar control.')
    

    st.title("Pancreatectomy Diabetes Predictor")

    if add_selectbox == 'An Individual':
        Age = st.number_input('Age', min_value=18, max_value=150, value=25)
        Gender = st.radio('Gender', ['M', 'F'])
        Ethnicity = st.radio('Ethnicity', ['White', 'BAME'])        
        BMI = st.number_input('BMI (kg/m²)', min_value=10.00, max_value=50.00)
        HTN = st.radio('Has Hypertension?', [True, False])        
        PreopOGTT = st.number_input('Pre-op Glucose (mmol/L)', min_value=1.0, max_value=50.0)
        PreopHbac = st.number_input('Pre-op Hba1c (mmol/mol)', min_value=1.0, max_value=500.0)
        PreOpAlbumin = st.number_input('Pre-op Albumin (g/L)', min_value=1.0, max_value=100.0)
        Procedure = st.radio('Procedure', ['Proximal', 'Distal/Central'])        
        DuodenalResection = st.number_input('Duodenal Resection Length (mm)', min_value=0.0, max_value=600.0)
        PancreasVolume = st.number_input('Pancreatic Resection Volume cm³', min_value=0.00000, max_value=1000000000.00000)
        Gastrectomy = st.radio('Gastrectomy', [True, False])        
 
        output=""

        input_dict = {'Age' : Age, 'Gender' : Gender, 'Ethnicity' : Ethnicity, 
                      'BMI' : BMI, 'HTN' : HTN, 'PreopOGTT' : PreopOGTT, 
                      'PreopHbac':PreopHbac, 'PreOpAlbumin': PreOpAlbumin,
                      'Procedure': Procedure, 'DuodenalResection': DuodenalResection,
                      'PancreasVolume': PancreasVolume, 'Gastrectomy': Gastrectomy 
                      }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            st.write('Your 1 year risk of developing diabetes is', round((predict(model=model, input_df=input_df))*100, 1), '%  \n A risk greater than 21.2% indicates a high risk of developing Diabetes.')



    if add_selectbox == 'Multiple Patients':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        st.write('Click here for a template to upload')

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            output = predict_model(estimator=model,data=data, probability_threshold=0.212)
            st.write(output)
            st.write('The score column is the predicted probability')

if __name__ == '__main__':
    run()
