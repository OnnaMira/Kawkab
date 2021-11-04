from __future__ import division, print_function
# coding=utf-8
import sys
import io
import random
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from matplotlib.figure import Figure
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from sklearn.metrics import mean_squared_error, mean_absolute_error
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc, roc_curve,accuracy_score,plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize

import seaborn as sns


from scipy import ndimage

from imblearn.over_sampling import SMOTE

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,request
from flask import Response,  Markup,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#load model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import seaborn as sn
from keras.optimizers.schedules import ExponentialDecay
from keras.models import load_model
from itertools import chain
import csv
import json
from PIL import Image
import base64
import io
import plotly
import plotly.express as px
################################################ DATA ##############################################
test = pd.read_csv('exoTest.csv')
#fig = px.line(test, x='date', y=['daily new cases','daily new recovered', 'daily new deaths'], title='Global daily new cases')
#fig.update_xaxes(rangeslider_visible=True)
#fig.show()

categ = {2: 1,1: 0}
#binarize labels
test.LABEL = [categ[item] for item in test.LABEL]
x_test = test.drop(["LABEL"],axis=1)
y_test = test["LABEL"]

def flux_graph(dataset, row, dataframe, planet):
    if dataframe:
        fig = plt.figure(figsize=(20,5), facecolor=	'black')
        ax = fig.add_subplot()
        ax.set_facecolor('black')
        ax.set_title(planet, color='white', fontsize=22)
        ax.set_xlabel('', color='white', fontsize=18)
        ax.set_ylabel('flux_' + str(row), color='white', fontsize=18)
        ax.grid(True,axis = 'y')
        flux_time = list(dataset.columns)
        flux_values = dataset[flux_time].iloc[row]
        ax.plot([i + 1 for i in range(dataset.shape[1])], flux_values, '#a84032')
        ax.tick_params(colors = 'white', labelcolor='white', labelsize=14)
       
        plt.savefig('static\img\saved_figure.png',dpi=fig.dpi,facecolor='auto',edgecolor= 'black')
        
        #encode
       # img = io.BytesIO()
       # plt.savefig(img, format='png')
       # img.seek(0)
       # plot_url = base64.b64encode(img.getvalue()).decode()
        #return(plot_url)





############################################# LOADING MODEL ###########################################

json_file = open('model99.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model99wights.h5")
print("Loaded model from disk")


 ############################################ FLASK APP#################################################


app = Flask(__name__)

########################################### Index #################################################
@app.route('/', methods=['POST','GET'])
def index():
    
   
    
    
    
    #if request.method == 'GET':
    return render_template('index.html', test='testing value' )
            
        
        
    
    
  
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        
            jsonData = request.get_json()
            print(jsonData)
            value = int(jsonData['starNum'] )
            
            
            print(value)
            #fig = px.line(test.iloc[value,], facet_col_wrap=2, width=1000, height=300,template = "plotly_dark",color="")
            flux_graph(dataset=test, row = int(value), dataframe=True,planet= 'Etoile {} '.format(int(value)))
            normalized = normalize(x_test)
            y_class_pred = (loaded_model.predict(x_test.iloc[[value]]) > 0.5).astype("int32")
        # model_plot = Markup('<img src="data:image/png;base64,{}" width="1200" height="300">'.format(plot_url))
            #print(model_plot)
            print(value)
        
            print('prediction :')
            print( y_class_pred )
            print('real')
            
            class_prediction = str(y_class_pred[0][0])
            real_class = str(y_test.iloc[value])
            print(real_class)
            dicty ={
                'pred' : class_prediction,
                #'imgjson': model_plot
                'real': real_class,
            }
            return jsonify(dicty)
            #return render_template('index.html',class_prediction =class_prediction  )
            
     

    

############################################ Accueil#################################################

@app.route('/Accueil', methods=['GET', 'POST'])
def index_acc():
    if request.method == 'POST':

        value = request.form['starNum'] 
       
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        print(value)
        
    # show the form, it wasn't submitted
    return render_template('Accueil.html')
    

############################################ Exo #################################################

@app.route('/Exoplanetes', methods=['GET', 'POST'])
def index_exo():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))
    # show the form, it wasn't submitted
    return render_template('Exoplanetes.html')

############################################ Contact #################################################
@app.route('/Contact', methods=['GET', 'POST'])
def index_con():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))
    # show the form, it wasn't submitted
    return render_template('Contact.html')

#####################################  App  ###########################################
if __name__ == '__main__':
    app.run(debug=True)

#####################################  DATA PREPROCESSING  ###########################################

def data_preprocessing():
    return 0


    ##################################### PNG ###########################################
@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

    ############################################ Get inpu@app.route('/background_process')
