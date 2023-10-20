import os
import warnings
import matplotlib
from flask import Flask, render_template, send_from_directory
import numpy as np
import pymysql
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import NearestNeighbors
import pandas as pd

app = Flask(__name__)
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

f_model_x = joblib.load('mergex.joblib')
f_model_y = joblib.load('mergey.joblib')

db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'user',
    'password': '1234',
    'database': 'esp32'
}

def get_data_from_mysql():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            query = "CALL Pivot();"
            cursor.execute(query)
            data = cursor.fetchall()
    finally:
        connection.close()

    return data

@app.route('/')
def index():
    data = get_data_from_mysql()
    data = np.array(data)
    plot_map(data)
    plot_table(data)
    return render_template('index.html')

df = pd.read_csv('data_f.csv')

rssi_values = df[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6']].values

predicted_x = f_model_x.predict(rssi_values)
predicted_y = f_model_y.predict(rssi_values)

k = 5
predicted_positions = np.column_stack((predicted_x, predicted_y))
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(predicted_positions)

knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(rssi_values)

def compute(data=None):
    if data is None:
        data = get_data_from_mysql()
    data = np.array(data)
    data = data.reshape(1, -1)

    predicted_x = f_model_x.predict(data)
    predicted_y = f_model_y.predict(data)
    real_time_position = np.column_stack((predicted_x, predicted_y))

    distances, indices = neigh.kneighbors(real_time_position)

    weights = 1 / distances
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    weights = weights[:, :, np.newaxis]

    estimated_position = np.sum(weights * predicted_positions[indices], axis=1)[0]

    distances2, indices2 = knn_model.kneighbors(data)

    weights2 = 1 / distances2
    weights2 = weights2 / np.sum(weights2, axis=1, keepdims=True)
    weights2 = weights2[:, :, np.newaxis]

    wknn_estimated_position = np.sum(weights2 * rssi_values[indices2], axis=1)[0]

    simple_knn_estimated_position = np.mean(rssi_values[indices2], axis=1)[0]
    return float(estimated_position[0]), float(estimated_position[1]), float(wknn_estimated_position[0]), float(wknn_estimated_position[1]), float(simple_knn_estimated_position[0]), float(simple_knn_estimated_position[1])
def create_map(xy):
    x_ranges = [(0, 3.3), (3.3, 6.6), (6.6, 10)]
    y_ranges = [(0, 2.5), (2.5, 5)]

    fig, ax = plt.subplots(figsize=(18, 9))
    x_value, y_value = xy[0], xy[1]
    for i in range(len(x_ranges)):
        x_min, x_max = x_ranges[i]
        y_min, y_max = y_ranges[1]

        if x_min <= x_value <= x_max and y_min <= y_value <= y_max:
            ax.fill_between([x_min, x_max], y_min, y_max, color='green', alpha=0.3)
    ax.plot([3.3, 3.3], [0, 5], 'b-.', linewidth=2)
    ax.plot([6.6, 6.6], [0, 5], 'b-.', linewidth=2)
    ax.plot([0, 10], [2.5, 2.5], 'b-.', linewidth=2)
    ax.plot(xy[0], xy[1], 'ro', label='GPR + WkNN')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('XY Coordinate Visualization')
    ax.grid(True)
    fig.legend(loc='upper right')

    return fig

@app.route('/map.png')
def plot_map(data=None):
    if data is None:
        data = get_data_from_mysql()
    xy = compute(data)
    fig = create_map(xy)
    output = os.path.join('static', 'map.png')
    fig.savefig(output)
    plt.close(fig)
    return send_from_directory('static', 'map.png')

def create_table(data=None):
    if data is None:
        data = get_data_from_mysql()
    data = np.array(data)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    column_labels = ["1", "2", "3", "4", "5", "6"]
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=data, colLabels=column_labels, colColours=["yellow"] * 6, loc="center",
                         bbox=[0, 0.8, 1.12, .3])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    return fig

@app.route('/table.png')
def plot_table(data=None):
    if data is None:
        data = get_data_from_mysql()
    fig = create_table(data)
    output = os.path.join('static', 'table.png')
    fig.savefig(output)
    plt.close(fig)
    return send_from_directory('static', 'table.png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
