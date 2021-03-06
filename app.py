from flask import Flask, render_template, request, redirect, Response, jsonify

import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import scipy.spatial.distance
import matplotlib.pyplot as plt
import json

app = Flask(__name__)


@app.route('/')
def index():
    df = pd.read_csv('data.csv')
    features = ['Overall', 'Balance', 'Stamina', 'Strength', 'HeadingAccuracy',
                'ShortPassing', 'LongPassing', 'Dribbling', 'BallControl', 'Acceleration',
                'SprintSpeed', 'Agility', 'ShotPower', 'Aggression', 'Jumping', 'Vision',
                'Composure', 'StandingTackle', 'SlidingTackle']
    df = df[features]
    df = df.dropna()

    random_sampling_results = random_sampling(df)
    k_means_sse_array, stratified_sampling_results = stratified_sampling(df)
    stratified_sampling_results = stratified_sampling_results[features]

    pca_no_sampling_results, pca_no_sampling_results_top_3_attributes, \
        pca_no_sampling_top_two_components = calculate_pca(df[features])
    pca_random_sampling_results, pca_random_sampling_results_top_3_attributes, \
        pca_random_sampling_top_two_components = calculate_pca(random_sampling_results)
    pca_stratified_sampling_results, pca_stratified_sampling_results_top_3_attributes, \
        pca_stratified_sampling_top_two_components = calculate_pca(stratified_sampling_results)

    pca_no_sampling_results_top_3_attributes_names = df.columns[pca_no_sampling_results_top_3_attributes]
    pca_random_sampling_results_top_3_attributes_names = df.columns[pca_random_sampling_results_top_3_attributes]
    pca_stratified_sampling_results_top_3_attributes_names = df.columns[
        pca_stratified_sampling_results_top_3_attributes]


    pca_no_sampling_projected_points = np.dot(stratified_sampling_results, pca_no_sampling_top_two_components.T)
    pca_random_sampling_projected_points = np.dot(stratified_sampling_results, pca_random_sampling_top_two_components.T)
    pca_stratified_sampling_projected_points = np.dot(stratified_sampling_results,
                                                      pca_stratified_sampling_top_two_components.T)

    mds_eucl = MDS(dissimilarity='euclidean')
    df_dimension_reduced = np.array(stratified_sampling_results[pca_stratified_sampling_results_top_3_attributes_names])
    mds_eucl.fit(df_dimension_reduced)
    mds_eucl_points = np.array(mds_eucl.embedding_ * 10, dtype=int)

    mds_corr = MDS(dissimilarity='precomputed')
    dissimilarity_matrix = scipy.spatial.distance.cdist(df_dimension_reduced, df_dimension_reduced,
                                                        metric='correlation')
    np.fill_diagonal(dissimilarity_matrix, np.zeros(len(dissimilarity_matrix)))
    dissimilarity_matrix = np.nan_to_num(dissimilarity_matrix)
    mds_corr.fit(dissimilarity_matrix)
    mds_corr_points = np.array(mds_corr.embedding_ * 10, dtype=int)


    data_frontend = dict()
    data_frontend["pca_no_sampling_variance_ratios"] = json.dumps(pca_no_sampling_results.tolist())
    data_frontend["pca_no_sampling_variance_ratios_cumsum"] = json.dumps(np.cumsum(pca_no_sampling_results).tolist())
    data_frontend["pca_no_sampling_results_top_3_attributes_names"] = json.dumps(
        pca_no_sampling_results_top_3_attributes_names.tolist())

    data_frontend["pca_random_sampling_variance_ratios"] = json.dumps(pca_random_sampling_results.tolist())
    data_frontend["pca_random_sampling_variance_ratios_cumsum"] = json.dumps(
        np.cumsum(pca_random_sampling_results).tolist())
    data_frontend["pca_random_sampling_results_top_3_attributes_names"] = json.dumps(
        pca_random_sampling_results_top_3_attributes_names.tolist())

    data_frontend["k_means_sse_array"] = json.dumps(k_means_sse_array)
    data_frontend["pca_stratified_sampling_variance_ratios"] = json.dumps(pca_stratified_sampling_results.tolist())
    data_frontend["pca_stratified_sampling_variance_ratios_cumsum"] = json.dumps(
        np.cumsum(pca_stratified_sampling_results).tolist())
    data_frontend["pca_stratified_sampling_results_top_3_attributes_names"] = json.dumps(
        pca_stratified_sampling_results_top_3_attributes_names.tolist())

    print(pca_stratified_sampling_results_top_3_attributes_names)


    data_frontend["pca_no_sampling_scree_plot_data"] = [{"factor": i + 1, "eigenvalue": pca_no_sampling_results[i],
         "cumulative_eigenvalue": np.cumsum(pca_no_sampling_results)[i]} for i in range(19)]
    data_frontend["pca_random_sampling_scree_plot_data"] = [{"factor": i + 1, "eigenvalue": pca_random_sampling_results[i],
         "cumulative_eigenvalue": np.cumsum(pca_random_sampling_results)[i]} for i in range(19)]
    data_frontend["pca_stratified_sampling_scree_plot_data"] = [{"factor": i + 1, "eigenvalue": pca_stratified_sampling_results[i],
         "cumulative_eigenvalue": np.cumsum(pca_stratified_sampling_results)[i]} for i in range(19)]

    data_frontend["pca_no_sampling_projected_points"] = [{"x": i[1], "y": i[0]} for i in pca_no_sampling_projected_points.tolist()]
    data_frontend["pca_random_sampling_projected_points"] = [{"x": i[1], "y": i[0]} for i in pca_random_sampling_projected_points.tolist()]
    data_frontend["pca_stratified_sampling_projected_points"] = [{"x": i[1], "y": i[0]} for i in pca_stratified_sampling_projected_points.tolist()]

    data_frontend["scatterplot_matrix_data"] = np.array(stratified_sampling_results[pca_stratified_sampling_results_top_3_attributes_names]).tolist()

    data_frontend["mds_eucl_data"] = [{"x": i[0], "y": i[1]} for i in mds_eucl_points]
    data_frontend["mds_corr_data"] = [{"x": i[0], "y": i[1]} for i in mds_corr_points]
    print(data_frontend["scatterplot_matrix_data"])


    data_frontend = {'chart_data': data_frontend}

    return render_template('index.html', data=data_frontend)


def random_sampling(data):
    data_arr = np.array(data)
    sample_size = int(0.05 * len(data_arr))
    random_indices = random.sample(range(len(data_arr)), sample_size)
    random_sampling_results = data_arr[random_indices]
    return random_sampling_results


def stratified_sampling(data):
    data_arr = np.array(data)
    sse_array = []
    for i in range(2, 20):
        km = KMeans(n_clusters=i)
        km.fit(data_arr)
        sse_array.append(km.inertia_)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2, 20), sse_array, marker='o', markeredgecolor='r', color='b')
    # ax.plot(range(2, 20), sse_array)
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow plot for K-Means clustering')
    plt.xticks(range(2, 20, 2))
    plt.show()

    km = KMeans(n_clusters=6)
    km.fit(data_arr)
    data['Label'] = km.labels_
    cluster_sizes = np.bincount(km.labels_)
    stratified_sampling_results = pd.DataFrame(columns=data.columns)
    for i in range(6):
        cluster_size = cluster_sizes[i]
        cluster_records = data[data['Label'] == i]
        sample_size = int(cluster_size * 0.05)
        stratified_sampling_results = pd.concat(
            [stratified_sampling_results, cluster_records.iloc[random.sample(range(cluster_size), sample_size)]])
    return sse_array, stratified_sampling_results


def calculate_pca(data):
    pca = PCA()
    pca.fit(data)
    pca_results = pca.explained_variance_ratio_
    loadings = np.sum(np.square(pca.components_), axis=0)
    indices_of_top_3_attributes = loadings.argsort()[-3:][::-1]
    top_two_components = pca.components_[:2]
    return pca_results, indices_of_top_3_attributes, top_two_components


if __name__ == '__main__':
    app.run(debug=True)
