import numpy as np
import scipy
import sklearn
from sklearn import metrics
import demo_data as dd
# H := |V| x |E| vertex-hyperedge incidence matrix
# De := diagonal matrix consisting of hyperedge degrees
# Dv := diagonal matrix consisting of weighted vertex degrees
# W := |E| x |E| diagonal matrix containing hyperedge weights

# Type of Vertices:
#   - Users
#   - Artists
#   - Tracks
#   - Albums
#   - Playlists

# Type of Relations:
#   - User-Track (user and track edge) [0,50|U|]
#   - User-Artist (user and artist edge) [0,50|U| + num_followed]
#   - Track-Album (album and its tracks edge)
#   - Album-Artist (artist and his/her albums edge)
#   - Track-Track (track similarity edge)
#   - Playlists (playlist)

# def add_edge(data, data_type):


# Data we have [n * (d * x)]
# d = 5
# most played tracks
# saved tracks
# most played artists
# followed artists
# saved albums


'''
input_data dims : [n * (d * x)] (x is variable)
idx1 : index of most played artists in input_data
idx2 : index of followed artists in input_data
returns set of unique artists
'''


def create_artist_vertices(input_data, idx1, idx2):
    artists = np.concatenate(
        (input_data[:, idx1, :], input_data[:, idx2, :]), axis=1)
    artists = np.ndarray.flatten(artists)
    return set(artists)


'''
input_data dims : [n * (d * x)] (x is variable)
idx1 : index of most played tracks in input_data
idx2 : index of liked tracks in input_data
returns set of unique tracks
'''


def create_track_vertices(input_data, idx1, idx2):
    tracks = np.concatenate(
        (input_data[:, idx1, :], input_data[:, idx2, :]), axis=1)
    tracks = np.ndarray.flatten(tracks)
    return set(tracks)


'''
input_data dims : [n * (d * x)] (x is variable)
idx : index of followed albums in input_data
returns set of unique albums
'''


def create_album_vertices(input_data, idx):
    albums = np.ndarray.flatten(input_data[:, idx, :], axis=1)
    return set(albums)


'''
tracks_mat : (n * d) matrix of audio features of each track
k : number of nearest neighbors
returns 2 d array of edges (track idx1, track idx2 , weight)
'''


def knn_track_similarity_edges(tracks_mat, k):
    features = np.array(tracks_mat)[:, 1:]
    sim_mat = metrics.pairwise.cosine_similarity(features)
    knn_idx = np.argsort(-1*sim_mat)[:, 1:k+1]
    knn_weights = -1*np.sort(-1*sim_mat)[:, 1:k+1]
    max_weight = np.amax(knn_weights)
    norm_knn_weight = -1*np.sort(-1*sim_mat)[:, 1:k+1]/max_weight
    edges = []
    for i in range(len(tracks_mat)):
        for neighbor in range(k):
            edges.append([tracks_mat[i, 0], tracks_mat[knn_idx[i, neighbor], 0],
                          norm_knn_weight[i, neighbor]])

    return edges


def remove_duplicates(mat):
    checker = set()
    new_mat = []
    for entry in mat:
        if entry[0] not in checker:
            checker.add(entry[0])
            new_mat.append(entry)

    return new_mat


song_features = remove_duplicates(dd.all_song_features)
print(knn_track_similarity_edges(np.array(song_features), 4))
