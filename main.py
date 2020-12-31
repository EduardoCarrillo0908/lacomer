from flask import Flask, render_template, request, jsonify
import time
import flask
import os
import json
from functions import prediction
from flask_cors import CORS, cross_origin
#app = Flask(__name__)
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def home():
    print("Hola mundo")
    return "hola mundo version 2"

@app.route("/get_users_products", methods=['POST','GET'])
def get_prods_users():
    if flask.request.method == 'POST':
        data = request.get_json(force=True)
        print(data)
        user = int(data['user'])
        store = data['store']
        modelo = data['model']
        try:
            dic = prediction.predict.user_products(user,store)
        except:
            print("error")
            dic=prediction.predict.get_promociones()
        return json.dumps(dic),200
    
@app.route("/get_promotions", methods=['POST','GET'])
def get_proms():
    if flask.request.method == 'POST':
        dic = prediction.predict.get_promociones()
        return json.dumps(dic),200
    
@app.route("/get_products", methods=['POST','GET'])
def get_products():
    if flask.request.method == 'POST':
        dic = prediction.predict.get_items()
        return json.dumps(dic),200

@app.route("/recommended_by_user",methods=['POST', 'GET'])
def get_recomm_by_user():
    if flask.request.method == 'POST':
        data = request.get_json(force=True)
        print(data)
        user = int(data['user'])
        store = data['store']
        modelo = data['model']
        print(modelo)
        if modelo == "knn":
            try:
                print ("Si hay knn")
                dic = prediction.predict.user_knn(user,store)
            except:
                print("no hay knn")
                dic = prediction.predict.user(user,store)
        else: 
            dic = prediction.predict.user(user,store)
        return json.dumps(dic),200

@app.route("/recommended_by_item",methods=['POST', 'GET'])
def similar_items():
    if flask.request.method == 'POST':
        print("Recomendando por item")
        data = request.get_json(force=True)
        item = data['item']
        store = data['store']
        dic = prediction.predict.similar_items(item,store)
        return json.dumps(dic),200

@app.route("/recommended_by_apriori",methods=['POST', 'GET'])
def apriori():
    if flask.request.method == 'POST':
        print("Recomendando por apriori")
        data = request.get_json(force=True)
        item = data['item']
        store = data['store']
        try:
            print ("apriori")
            dic = prediction.predict.apriori_model(item,store)
        except:
            print ("no apriori")
            dic = {}
        return json.dumps(dic),200
     
if __name__ == "__main__":
    #app.run(debug=True, port=5050)
    app.run(host='0.0.0.0', port=5050)
    #app.run(host='0.0.0.0')
    

