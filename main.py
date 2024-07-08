from flask import Flask, render_template, request, redirect, Response, jsonify, json
import csv
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator

app = Flask(__name__)

def preprocess_data(variable_flag = False):
    data = pd.read_csv("static/data.csv")
    label_encoder = preprocessing.LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])
    data.drop(columns=['id'], inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    sse = []
    labels = {}

    for k in range(1,6):

        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
        labels[k] = kmeans.labels_
        sse.append(kmeans.inertia_)
    
    kl = KneeLocator(range(1, 6), sse, curve="convex", direction="decreasing" )
    labels = labels[kl.elbow]

    if variable_flag:
        distane_matrix = 1 - np.abs(data.corr())
        return scaled_data, labels, distane_matrix, data.columns

    return scaled_data,labels


@app.route("/mds", methods = ['POST', 'GET'])
def mds():

    scaled_data,labels = preprocess_data()

    mds = MDS(n_components = 2)
    mds_data = mds.fit_transform(scaled_data)

    data = {
        "mds": mds_data.tolist(),
        "labels":labels.tolist()
    }

    data_json = json.dumps(data)

    return render_template("mds.html", data=data_json)


@app.route("/pcp", methods=['GET', 'POST'])
def pcp():

    data = pd.read_csv("static/data.csv")
    label_encoder = preprocessing.LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])
    data.drop(columns=['id'], inplace=True)

    normal_data = data.to_dict()

    print("type-->",type(data))

    data = {
        "normal_data": normal_data
    }

    data_json = json.dumps(data)

    return render_template("pcp.html", data=data_json)

@app.route("/mds_var", methods=['POST', 'GET'])
def mds_var():

    scaled_data,labels, distance_matrix,columns = preprocess_data(variable_flag=True)

    # print("distance_matrix-->",distance_matrix)
    column_names = columns.tolist()

    scaled_data_T = scaled_data.T
    # print("Shape of scaled_data_transpose",scaled_data_T.shape)

    mds = MDS(n_components = 2, dissimilarity="precomputed")
    mds_data = mds.fit_transform(distance_matrix)

    data = {
        "mds": mds_data.tolist(),
        "labels":labels.tolist(),
        "column_names":column_names
    }

    data_json = json.dumps(data)

    print(data_json)

    return render_template("mds_var.html", data=data_json)

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
