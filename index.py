from flask import Flask, render_template, request

from sklearn import datasets
from sklearn import tree
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET','POST'])
def result():
    
    test_data = [39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States"]
    
    new_data = [];
    new_data.append(request.form["age"])
    new_data.append(request.form["fnlwgt"])
    new_data.append(request.form["education-num"])
    new_data.append(request.form["capital-gain"])
    new_data.append(request.form["capital-loss"])
    new_data.append(request.form["hours-per-week"])
    
    workclass = request.form["workclass"]
    workclassList = ["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"]

    for i in range(len(workclassList)) :
        if (workclassList[i] == workclass) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    education = request.form["education"]
    educationList = ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-Grad", "Masters", "Preschool", "Prof-School", "Some-college"]

    for i in range(len(educationList)) :
        if (educationList[i] == education) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    marital = request.form["marital-status"]
    maritalList = ["Divorced","Married-AF-Spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"]
    
    for k in range(len(maritalList)) :
        if (maritalList[k] == marital) :
            new_data.append("1")
        else :
            new_data.append("0")

    occupation = request.form["occupation"]
    occupationList = ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-speciality", "Protective-serv", "Sales", "Tech-support", "Transport-moving"]
    
    for i in range(len(occupationList)) :
        if (occupationList[i] == occupation) :
            new_data.append("1")
        else :
            new_data.append("0")

    relationship = request.form["relationship"]
    relationshipList = ["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
    
    for i in range(len(relationshipList)) :
        if (relationshipList[i] == relationship) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    race = request.form["race"]
    raceList = ["American-Eskimo", "Accian-Pac-Islander", "Black", "Other", "White"]
    
    for i in range(len(raceList)) :
        if (raceList[i] == race) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    sex = request.form["sex"]
    sexList = ["Male", "Female"]
    
    for i in range(len(sexList)) :
        if (sexList[i] == sex) :
            new_data.append("1")
        else :
            new_data.append("0")

    country = request.form["native-country"]
    countryList = ["Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos","Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"]
    
    for i in range(len(countryList)) :
        if (countryList[i] == country) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    new_data = np.asarray(new_data)
    new_data = np.reshape(new_data, (1,-1))
    clf = joblib.load('model/clf.pkl')
    prediction = clf.predict(new_data)

    if (prediction[0] == 0.0) :
        return render_template('result.html', result="<=50k",
                                _age=request.form["age"],
                                _workclass=request.form["workclass"],
                                _fnlwgt=request.form["fnlwgt"],
                                _education=request.form["education"],
                                _education_num=request.form["education-num"],
                                _marital_status=request.form["marital-status"],
                                _occupation=request.form["occupation"],
                                _relationship=request.form["relationship"],
                                _race=request.form["race"],
                                _sex=request.form["sex"],
                                _capital_gain=request.form["capital-gain"],
                                _capital_loss=request.form["capital-loss"],
                                _hours_per_week=request.form["hours-per-week"],
                                _native_country=request.form["native-country"])
    else :
        return render_template('result.html', result=">50k",
                                _age=request.form["age"],
                                _workclass=request.form["workclass"],
                                _fnlwgt=request.form["fnlwgt"],
                                _education=request.form["education"],
                                _education_num=request.form["education-num"],
                                _marital_status=request.form["marital-status"],
                                _occupation=request.form["occupation"],
                                _relationship=request.form["relationship"],
                                _race=request.form["race"],
                                _sex=request.form["sex"],
                                _capital_gain=request.form["capital-gain"],
                                _capital_loss=request.form["capital-loss"],
                                _hours_per_week=request.form["hours-per-week"],
                                _native_country=request.form["native-country"])

if __name__ == '__main__' :
    app.run(debug = True)