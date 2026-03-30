import streamlit as st
import pandas as pd
from model import predict_logistic

features,target,X,Y,X_train,X_test,Y_train,Y_test,scaler,model,Y_pred=predict_logistic()

st.header("Employee Termination or Active Prediction using Logistic Regression.")
st.subheader("hek")

st.sidebar.header(
    "Employee Status Prediction"
)

PerfScoreID =st.sidebar.slider(
    "Performance Score",
    value = 3,
    min_value=0,
    max_value=5,
    step=1

)
SalaryID= st.sidebar.slider(
    "Salary",
    value=10000,
    min_value=1000,
    max_value=600000,
    step=10
)
PositionID= st.sidebar.slider(
    "PositionID",
    value=5,
    min_value=0,
    max_value=30,
    step=1
)
EngagementSurvey= st.sidebar.slider(
    "Engagement Survey",
    value=5.0,
    min_value=0.0,
    max_value=30.0,
    step=0.1
)
EmpSatisfaction= st.sidebar.slider(
    "Emp Satisfaction",
    value=3,
    min_value=0,
    max_value=5,
    step=1
)
SpecialProjectsCount= st.sidebar.slider(
    "Special Projects Count'",
    value=5,
    min_value=1,
    max_value=30,
    step=1
)
DaysLateLast30= st.sidebar.slider(
    "Late Days",
    value=5,
    min_value=0,
    max_value=30,
    step=1
)
Absences= st.sidebar.slider(
    "Absences",
    value=5,
    min_value=0,
    max_value=30,
    step=1
)
if st.button("Predict Status"):
    input_data=pd.DataFrame([[
        PerfScoreID,SalaryID,PositionID,EngagementSurvey,EmpSatisfaction,SpecialProjectsCount,DaysLateLast30,Absences
    ]],columns=features)
    input_scaler=scaler.transform(input_data)
    prediction=model.predict(input_scaler)

    if prediction[0]==0:
        st.success("Employee is likely to be active in future 👍👌😊")
        result='Active'
    else:
        st.warning("Employee is likely to be terminated in futre..")
        result='Terminated'
    st.metric(
        "Employee Status",
        value=result
    )

    