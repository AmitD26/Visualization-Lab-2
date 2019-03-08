from flask import Flask

import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

app = Flask(__name__)

df = pd.read_csv('data.csv')
features = ['Overall', 'Balance', 'Stamina', 'Strength', 'HeadingAccuracy',
            'ShortPassing', 'LongPassing', 'Dribbling', 'BallControl', 'Acceleration',
            'SprintSpeed', 'Agility', 'ShotPower', 'Aggression', 'Jumping', 'Vision',
            'Composure', 'StandingTackle', 'SlidingTackle']
df = df[features]
random_sampling_results = []
stratified_sampling_results = []


@app.route('/')
def index():
    # print(df.columns)
    # random_sampling()
    stratified_sampling()
    return 'hello'


def random_sampling():
    global df
    global random_sampling_results
    df = df.dropna()
    df_arr = np.array(df)
    sample_size = int(0.25 * len(df_arr))
    random_indices = random.sample(range(len(df_arr)), sample_size)
    random_sampling_results = df_arr[random_indices]
    print(len(df_arr))
    print(len(random_sampling_results))


def stratified_sampling():
    global df
    global stratified_sampling_results
    df = df.dropna()
    df_arr = np.array(df)
    print(len(df_arr))
    sse_array = []
    for i in range(2, 20):
        km = KMeans(n_clusters=i)
        km.fit(df_arr)
        sse_array.append(km.inertia_)
    print(sse_array)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2, 20), sse_array, marker='o', markeredgecolor='r', color='b')
    # ax.plot(range(2, 20), sse_array)
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow plot for K-Means clustering')
    plt.xticks(range(2, 20, 2))
    print("End of plotElbow")
    plt.show()


if __name__ == '__main__':
    app.run(debug=True)
