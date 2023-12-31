from django.shortcuts import render
from . import fake_model
from . import ml_predict

def home(request):
    return render(request, 'index.html')

def results(request):
    pclass = int(request.GET['pclass'])  # note .GET[]
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embarked = int(request.GET['embarked'])
    title = int(request.GET['title'])
    #user_input_age += " is the word"
    #prediction = fake_model.fake_predict(user_input_age)
    print(pclass,sex,age,sibsp,parch,fare,embarked,title)
    prediction = ml_predict.prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title)
    context = {'prediction' : prediction,
               }
    return render(request, 'results.html', context)