import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from pandas import read_csv
import joblib

@st.cache_resource
def load_data():
    return read_csv('X_test.csv'), read_csv('Y_test.csv')

@st.cache_resource
def load_model():
    return {
        'Ada Boost': joblib.load('AdaBoost.joblib'),
        'Extra Trees': joblib.load('ExtraTrees.joblib'),
        'Gradient Boosting': joblib.load('GradientBoosting.joblib'),
        'Random Forest': joblib.load('RandomForest.joblib')
    }


X_test, Y_test = load_data()
models = load_model()
st.title('Prognostic value of neutrophil-to-lymphocyte ratio in septic patients with liver cirrhosis')

mortalities = ['mortality_7d', 'mortality_28d', 'mortality_90d', 'mortality_1y']
titles = ['7 days', '28 days', '90 days', '1 years']
fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (20, 8))
for i in range(len(mortalities)):
    y_test = Y_test[mortalities[i]]
    for name in models:
        model = models[name]
        RocCurveDisplay.from_estimator(estimator = model, X = X_test, y = y_test, ax = axes[i], name = name)
    axes[i].set_title(titles[i])
st.pyplot(fig)



#ctrl + shift + p
#