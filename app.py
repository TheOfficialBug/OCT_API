from flask import Flask,render_template,request,jsonify,flash,redirect,url_for
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf

from classfication import pred
app=Flask(__name__,static_folder="assets")

 

app.secret_key = "secret key"
app.config['SERVER_NAME'] = 'localhost:5000'
@app.route('/')
def home():
    return "hello"



@app.route('/predection',methods = ['GET','POST'])
def predection():
    if request.method=="POST":
        file=request.files["image"]
        file.save(file.filename)
        img=tf.keras.utils.load_img(file.filename)
        state=pred(img)
        os.remove(file.filename)
        return str(state)

        


   






    
if __name__=="__main__":
    app.run(debug=True)




