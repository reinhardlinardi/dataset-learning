from flask import Flask, render_template, request
import numpy as np
from sklearn import datasets
from sklearn import tree
import pandas as pd
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET','POST'])
def result():
    '''
    test_data = [39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States"]
    
    new_data = [];
    new_data.append(request.form["age"])
    new_data.append(request.form["fnlwgt"])
    new_data.append(request.form["education-num"])
    new_data.append(request.form["capital-gain"])
    new_data.append(request.form["capital-loss"])
    new_data.append(request.form["hours-per-week"])
    
    workclass = request.form["workclass"]
    
    for i in range (8) :
        if (((i == 0) and (workclass == "Federal-gov")) or ((i == 1) and (workclass == "Local-gov")) or ((i == 2) and (workclass == "Never-worked")) or ((i == 3) and (workclass == "Private")) or ((i == 4) and (workclass == "Self-emp-inc")) or ((i == 5) and (workclass == "Self-emp-not-inc")) or ((i == 6) and (workclass == "State-gov")) or ((i == 7) and (workclass == "Without-pay"))) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    education = request.form["education"]
    educationList = ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-Grad", "Masters", "Preschool", "Prof-School", "Some-college"]
    for j in range(16) :
        if (educationList[j] == education) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    maritalList = ["Married-AF-Spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"]
    marital = request.form["marital"]
    for k in range(len(maritalList)) :
        if (maritalList[k] == marital) :
            new_data.append("1")
        else :
            new_data.append("0")

    occupationList = ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-speciality", "Protective-serv", "Sales", "Tech-support", "Transport-moving"]
    occupation = request.form["occupation"]
    for l in range(len(occupationList)) :
        if (occupationList[l] == occupation) :
            new_data.append("1")
        else :
            new_data.append("0")

    relationshipList = ["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
    relationship = request.form["relationship"]
    for i in range(len(relationshipList)) :
        if (relationshipList[i] == relationship) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    raceList = ["American-Eskimo", "Accian-Pac-Islander", "Black", "Other", "White"]
    race = request.form["race"]
    for i in range(len(raceList)) :
        if (raceList[i] == race) :
            new_data.append("1")
        else :
            new_data.append("0")
    
    sexList = ["Male", "Female"]
    sex = request.form["sex"]
    for i in range(len(sexList)) :
        if (sexList[i] == sex) :
            new_data.append("1")
        else :
            new_data.append("0")

    countryList = ["Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos","Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"]
    country = request.form["country"]
    for i in range(len(countryList)) :
        if (countryList[i] == country) :
            new_data.append("1")
        else :
            new_data.append("0")

    
    
    #new_data.append()
    new_data = np.asarray(new_data)
    new_data = np.reshape(new_data, (1,-1))
    clf = joblib.load('clf.pkl')
    prediction = clf.predict(new_data)
    return render_template('result.html', result = type(prediction))
    '''
    return render_template('result.html')

if __name__ == '__main__' :
    app.run(debug = True)