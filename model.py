import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('HR_Dataset Refresh.csv')

def predict_logistic():
    features = ['PerfScoreID', 'Salary', 'PositionID', 'EngagementSurvey', 'EmpSatisfaction','SpecialProjectsCount', 'DaysLateLast30','Absences']
    target = 'Termd'

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, 
        stratify=Y # for equal distribution of target values
    )

    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train) # Xi-mean/sd
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver='liblinear', class_weight='balanced', random_state=42
    )
    model.fit(X_train_scale, Y_train)
    Y_pred = model.predict(X_test_scale)
    return features,target,X,Y,X_train,X_test,Y_train,Y_test,scaler,model,Y_pred