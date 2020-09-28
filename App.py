# -*- coding: utf-8 -*-
"""
Created on Sun Sept 6 09:20:38 2020

@author: Dr Aditya Borakati
"""
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import math
import markdown
import bokeh
import bokeh.models
from bokeh.plotting import figure, output_file, show
from bokeh.core.properties import Enum
from bokeh.models import HoverTool, Band, ColumnDataSource
import collections
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import sem
import json



observed=json.loads('{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":0,"11":0,"12":0,"13":0,"14":0,"15":0,"16":0,"17":0,"18":0,"19":0,"20":0,"21":0,"22":0,"23":0,"24":0,"25":1,"26":0,"27":0,"28":0,"29":0,"30":0,"31":0,"32":0,"33":0,"34":0,"35":0,"36":0,"37":0,"38":0,"39":0,"40":1,"41":1,"42":0,"43":0,"44":1,"45":1,"46":1,"47":0,"48":0,"49":0,"50":0,"51":0,"52":1,"53":1,"54":0,"55":0,"56":0,"57":1,"58":0,"59":0,"60":1,"61":0,"62":0,"63":0,"64":0,"65":0,"66":0,"67":0,"68":0,"69":0,"70":1,"71":0,"72":0,"73":1,"74":0,"75":1,"76":0,"77":0,"78":0,"79":1,"80":1,"81":1,"82":0,"83":0,"84":0,"85":0,"86":0,"87":0,"88":0,"89":0,"90":1,"91":1,"92":0,"93":0,"94":0,"95":0,"96":0,"97":0,"98":0,"99":0,"100":0,"101":1,"102":0,"103":0,"104":1,"105":1,"106":0,"107":0,"108":0,"109":0,"110":0,"111":0,"112":0,"113":1,"114":0}')
observed=pd.DataFrame(observed, index=[0])
observed=observed.T

probs=json.loads('{"0":0.128,"1":0.2577,"2":0.1331,"3":0.1364,"4":0.1491,"5":0.2713,"6":0.232,"7":0.1296,"8":0.1506,"9":0.1253,"10":0.1314,"11":0.1565,"12":0.1671,"13":0.145,"14":0.1524,"15":0.2124,"16":0.218,"17":0.1245,"18":0.1438,"19":0.124,"20":0.1344,"21":0.394,"22":0.1732,"23":0.16,"24":0.2625,"25":0.6151,"26":0.5553,"27":0.1601,"28":0.1828,"29":0.1476,"30":0.142,"31":0.1385,"32":0.1321,"33":0.123,"34":0.1575,"35":0.1697,"36":0.1319,"37":0.2008,"38":0.1596,"39":0.1476,"40":0.2725,"41":0.4619,"42":0.2276,"43":0.2852,"44":0.3571,"45":0.5212,"46":0.3533,"47":0.1871,"48":0.4674,"49":0.1391,"50":0.1467,"51":0.1315,"52":0.4053,"53":0.2762,"54":0.2759,"55":0.4032,"56":0.2588,"57":0.1955,"58":0.3764,"59":0.2296,"60":0.617,"61":0.1912,"62":0.1314,"63":0.211,"64":0.3139,"65":0.1503,"66":0.2212,"67":0.2026,"68":0.2061,"69":0.1575,"70":0.2151,"71":0.335,"72":0.283,"73":0.4436,"74":0.1385,"75":0.4283,"76":0.1209,"77":0.1318,"78":0.273,"79":0.4248,"80":0.3872,"81":0.1738,"82":0.2333,"83":0.3199,"84":0.3619,"85":0.1189,"86":0.285,"87":0.1864,"88":0.2264,"89":0.2212,"90":0.4354,"91":0.2156,"92":0.1536,"93":0.1507,"94":0.2081,"95":0.1665,"96":0.2502,"97":0.1867,"98":0.141,"99":0.1374,"100":0.1936,"101":0.4096,"102":0.1406,"103":0.2334,"104":0.3713,"105":0.3602,"106":0.1318,"107":0.1693,"108":0.1221,"109":0.1414,"110":0.1691,"111":0.1707,"112":0.1504,"113":0.3082,"114":0.1195}')
probs=pd.DataFrame(probs, index=[0])
probs=probs.T

weight=json.loads('{"0":15.0498123168945,"1":11.9054279327393,"2":10.1891422271729,"3":7.48856735229492,"4":6.43929147720337,"5":6.01290369033814,"6":5.09448528289795,"7":4.78982830047607,"8":4.65074443817139,"9":4.26376628875732,"10":3.65172553062439,"11":3.48581409454346,"12":3.30386543273926,"13":3.28845191001892,"14":3.06338953971863,"15":2.84500861167908,"16":2.58615159988403,"17":1.89162790775299}')
weight=pd.DataFrame(weight, index=[0])
weight=weight.T
label=json.loads('["Pre-operative Random Glucose","Pre-operative Random Glucose x Pre-operative Hba1c^3","Pre-operative Hba1c^3","Distal Procedure x Pancreatic Resection Volume^3","Pancreatic Resection Volume^3","Hypertension","Pre-operative Hba1c^3 x Pancreatic Resection Volume^3","Small Bowel Resection Length","Age","Body Mass Index","Black or Minority Ethnicity","Pre-operative Albumin x Pancreatic Resection Volume^3","Distal Procedure","Pancreatic Resection Volume^3 x Body Mass Index","Male Gender","Pancreatogastrostomy","Pre-operativeAlbumin","Benign Indication"]')
label=pd.DataFrame(label)

weight.reset_index(drop=True, inplace=True)
label.reset_index(drop=True, inplace=True)
featimpdf=pd.concat([weight, label], axis=1)
featimpdf.columns=['Weight','Label']

fpr, tpr, thresholds = roc_curve(observed, probs)

ci = 1.96 * np.std(tpr)/math.sqrt(38)

df=pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'threshold': thresholds})
df['specificity']=np.round((1-fpr)*100, 1)
df['sensitivity']=np.round(tpr*100, 1)
def _markdown(text):
    return bokeh.models.widgets.markups.Div(
        text=markdown.markdown(text), sizing_mode="stretch_width"
    )


def about_panel():
    text = """
## About

This app has been created to predict the risk of developing diabetes mellitus in one year after having a partial pancreatectomy. A partial pancreatectomy is a surgical procedure to remove part of your pancreas, a leaf shaped organ in your abdomen which is important in blood sugar control.

It has been developed using a database of 380 patients undergoing partial pancreatectomy at the Royal Free Hospital, London, UK. Of these patients 75 (19.7%) developed diabetes within one year of their surgery.

The underlying model uses the [Extreme Gradient Boosting](https://xgboost.readthedocs.io/en/latest/) (XGBoost) algorithm to predict risk of diabetes at one year.

"""
    return bokeh.models.Panel(child=_markdown(text), title="About")

def copyright_panel():
    text = """
## Copyright and License

<p style="font-family:Arial, sans-serif;">Copyright © A Borakati 2020 <br /> Licensed under the <a href="https://creativecommons.org/licenses/by/4.0/legalcode">Creative Commons Attribution 4.0 International Public License</a> <br /> Source Code available <a href="https://github.com/ABorakati/PancreatectomyPredictor">here</a><br /> Contact: <a href="mailto: a.borakati@doctors.org.uk">a.borakati@doctors.org.uk</a></p>

"""
    return bokeh.models.Panel(child=_markdown(text), title="Copyright and Licence")

def roc_panel():
    p = bokeh.plotting.figure(sizing_mode="stretch_width", height=400)
    
    TOOLS = "pan,zoom_in,zoom_out,reset,save"
    p = figure(title='Receiver Operating Characteristic', width=500, height=500,
           x_range=(-0.0025, 1), y_range=(0,1.0025), tools=TOOLS)
    p.title.align = 'center'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    p.ray(x=[0], y=[0], length=math.sqrt(2), angle=math.pi/4, line_width=3,
      line_color="black",line_dash=[20, 10])
    p.legend.location = 'bottom_right'
    p.xaxis.axis_label = 'False Positive Rate'
    p.yaxis.axis_label = 'True Positive Rate'
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.xaxis.major_tick_line_color = None  # turn off x-axis minor ticks

    p.xaxis.axis_line_color = "gainsboro"
    p.yaxis.axis_line_color = "gainsboro"
    p.yaxis.major_tick_line_color = None  # turn off x-axis minor ticks

    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

    lower= tpr-ci
    upper = tpr+ci
    source = ColumnDataSource(df)
    p.line(x="fpr", y="tpr", source=source, legend='AUC: 89.7 (95% CI 83.5-96.0)', line_width=3,
       )
    p.varea(x=fpr, y1=lower, y2=upper,fill_color="gainsboro", fill_alpha=0.4,
        legend_label="95% Confidence Interval") 
    p.legend.click_policy = "hide"
    p.legend.location = "bottom_right"

    p.add_tools(HoverTool(mode='vline',line_policy='nearest',
    tooltips = [
    ("Sensitivity","@sensitivity{1.1}"),
    ("Specificity","@specificity{1.1}"),
    ("Probability Threshold", "@threshold")
    ]
    ))

    
    
    text = """
<br/>Optimum probability threshold is <b>26.7%</b>    
<br/>Hover over chart to get sensitivity and specifity values at different points

    """    
    
    column = bokeh.layouts.Column(
        children=[p, _markdown(text),], sizing_mode="stretch_width"
    )
    

    return bokeh.models.Panel(child=column, title="ROC")

def variables_panel():



   bar=ColumnDataSource(featimpdf)


   p=figure(y_range=featimpdf['Label'],x_range=(0,16.0), title="Variable Importance to Model", width=750, height=375,   
   tools='pan,zoom_in, zoom_out, reset')
   p.hbar(right='Weight', y='Label', source=bar, height=0.9)
   p.xaxis.axis_label = 'Gini Importance to Model (%)'
   p.yaxis.axis_label = 'Variables'
   p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
   p.xaxis.major_tick_line_color = None  # turn off x-axis minor ticks

   p.xaxis.axis_line_color = "gainsboro"
   p.yaxis.axis_line_color = "gainsboro"
   p.yaxis.major_tick_line_color = None  # turn off x-axis minor ticks

   p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
   p.title.align = 'center'
   p.xgrid.grid_line_color = None
   p.ygrid.grid_line_color = None
   p.add_tools(HoverTool(tooltips=[("Weight", "@Weight"), ("Variable", "@Label")], point_policy='follow_mouse'))
   text = """
<br/>Hover over chart to get values
   """
   column = bokeh.layouts.Column(
        children=[p, _markdown(text),], sizing_mode="stretch_width"
    )   
   return bokeh.models.Panel(child=column, title="Variables")

def accuracy_panel():

    text = """
## Diagnostic Accuracy Metrics

<table class="table table-striped table-bordered">
<thead>
<tr>
<th>Metric</th>
<th>Median % (95 % Confidence Interval)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Sensitivity</td>
<td>86.9 (69.5-100.0)</td>
</tr>
<tr>
<td>Specificity</td>
<td>84.8 (58.7-95.7)</td>
</tr>
<tr>
<td>Positive Predictive Value</td>
<td>58.3 (37.7-82.6)</td>
</tr>
<tr>
<td>Negative Predictive Value</td>
<td>96.4 (92.3 -100.0)</td>
</tr>
<tr>
<td>Diagnostic Accuracy</td>
<td>85.2 (66.1-92.2)</td>
</tr>
<tr>
<td>Area under the Curve (C-statistic)</td>
<td>89.7 (79.3-94.3)</td>
</tr>
</tbody>
</table>

Values at Optimum Probability Threshold of <b>26.7%</b>
    """
    return bokeh.models.Panel(child=_markdown(text), title="Accuracy")
model = load_model('xgb3')

def disclaimer_panel():
    text = """
## Disclaimer

The model presented here has been developed to be as accurate as possible at the time of release. However, there is no guarantee or warranty of accuracy.
The authors and developer are in no way liable for the outcomes resulting from use of this model.
This model does not replace clinical judgement by a medical professional and the results of any predictions should be discussed with your clinician.
No data entered into this application is stored.

"""
    return bokeh.models.Panel(child=_markdown(text), title="Disclaimer")



def predict(model, input_df):
    predict_df = predict_model(estimator=model, data=input_df)
    predictions = predict_df['Score'][0]
 ###   percentrisk = print("%.2f%%" % (predictions * 100.0))
    return predictions


st.set_option('deprecation.showfileUploaderEncoding', False)


def run():
    st.markdown('<h1 style="font-family: Arial, sans-serif; text-align:center; font-weight:normal;">Pancreatectomy Diabetes Predictor</h1>',unsafe_allow_html=True)

    add_selectbox = st.selectbox(
        "Who are you predicting for?",
        ("An Individual", "Multiple Patients"))




    if add_selectbox == 'An Individual':
        Age = st.slider('Age', min_value=18, max_value=100, value=50)
        Gender = st.radio('Gender', ['M', 'F'])
        Ethnicity = st.radio('Ethnicity', ['White', 'BAME'])
        BMI = st.number_input('BMI (kg/m²)', min_value=10.00,
                              max_value=50.00, value=20.0)
        HTN = st.radio('Has Hypertension?', [True, False])
        PreopOGTT = st.number_input(
            'Pre-op Glucose (mmol/L)', min_value=1.0, max_value=50.0, value=7.0)
        PreopHbac = st.number_input(
            'Pre-op Hba1c (mmol/mol)', min_value=1.0, max_value=500.0, value=28.0)
        PreOpAlbumin = st.number_input(
            'Pre-op Albumin (g/L)', min_value=1.0, max_value=100.0, value=40.0)
        Indication = st.radio('Indication for Procedure', [
                              'Benign', 'Malignant', 'Pancreatitis', 'NET'])
        Procedure = st.radio('Procedure', ['Proximal', 'Distal'])
        DuodenalResection = st.number_input(
            'Small Bowel Resection Length (mm)', min_value=0.0, max_value=600.0)
        PancreasVolume = st.number_input(
            'Pancreatic Resection Volume cm³', min_value=0.00000, max_value=1000000000.00000)
        Pancreatojejunostomy = st.radio('Pancreatojejunostomy', [True, False])

        output = ""

        input_dict = {'Age': Age, 'Gender': Gender, 'Ethnicity': Ethnicity,
                      'BMI': BMI, 'HTN': HTN, 'PreopOGTT': PreopOGTT,
                      'PreopHbac': PreopHbac, 'PreOpAlbumin': PreOpAlbumin, 'Indication': Indication,
                      'Procedure': Procedure, 'DuodenalResection': DuodenalResection,
                      'PancreasVolume': PancreasVolume, 'PJ': Pancreatojejunostomy
                      }
        input_df = pd.DataFrame([input_dict])
        
        if st.button("Predict"):
            st.write('Your 1 year risk of developing diabetes is', round((predict(model=model, input_df=input_df))
                                                                         * 100, 1), '%  \n A risk greater than 26.7% indicates a high risk of developing Diabetes.')


    

    if add_selectbox == 'Multiple Patients':

        file_upload = st.file_uploader(
            "Upload Excel file to predict risk for Multiple Patients", type=["xlsx"])
        st.write('[Click here](https://github.com/ABorakati/PancreatectomyPredictor/blob/master/Template.xlsx) for a template to upload')

        if file_upload is not None:
            data = pd.read_excel(file_upload)
            output = predict_model(
                estimator=model, data=data, probability_threshold=0.267)
            st.write(output)
            st.write('The score column is the predicted probability')


    tabs = bokeh.models.Tabs(
        tabs=[
            about_panel(),
            roc_panel(),
            variables_panel(),
            accuracy_panel(),
            disclaimer_panel(),
            copyright_panel()
        ]
    )
    st.bokeh_chart(tabs)

hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == '__main__':
    run()
