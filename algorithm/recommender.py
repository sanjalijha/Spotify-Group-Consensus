import numpy as np
import scipy
import sklearn
from sklearn import metrics
from demo_data import data, album_track_data, artist_albums_data, all_song_features

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

# Data we have [n * (d * x)]
# d = 7
# most played tracks
# favorite tracks
# most played artists
# followed artists
# favorite albums
# get artist's albums

no_of_users = len(data)
no_of_tracks = 0
no_of_albums = 0
no_of_artists = 0

no_of_e3 = 0
no_of_e7 = 0
no_of_e8 = 0
no_of_e9 = 0


def user_track(data, x):
    no_of_edges = len(x)*50*no_of_users
    edges = [['', ''] for i in range(no_of_edges)]
    ind = 0
    for i in x:
        for p in range(no_of_users):
            tracks = data[p][i]
            for t in range(len(tracks)):
                edges[ind] = [str(p), tracks[t]]
                ind += 1
    return edges


def album_track_size(data):
    size = 0
    for p in range(no_of_users):
        size += len(data[p])
    return size


def create_vertices(data, x):
    vertices = []
    for i in x:
        for p in range(no_of_users):
            vertices = vertices + data[p][i]
    return set(vertices)


def create_indices(start_ind, input_set):
    hashmap = {}
    ind = start_ind
    for i in input_set:
        hashmap[i] = ind
        ind += 1
    return hashmap


tracks_set = create_vertices(data, [0, 1])
no_of_tracks = len(tracks_set)
tracks_start_ind = no_of_users
tracks_indices = create_indices(tracks_start_ind, tracks_set)
# print(tracks_indices)

albums_set = create_vertices(data, [4])
no_of_albums = len(albums_set)
albums_start_ind = tracks_start_ind + no_of_tracks
albums_indices = create_indices(albums_start_ind, albums_set)
# print(albums_indices)

artists_set = create_vertices(data, [2, 3])
no_of_artists = len(artists_set)
artists_start_ind = albums_start_ind + no_of_albums
artists_indices = create_indices(artists_start_ind, artists_set)
# print(artists_indices)

e3 = user_track(data, [0, 1])
no_of_e3 = len(e3)
print(no_of_e3)

no_of_e7 = album_track_size(album_track_data)
print(no_of_e7)
# def add_edge(data, data_type):


# Data we have [n * (d * x)]
# d = 5
# most played tracks
# saved tracks
# most played artists
# followed artists
# saved albums


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


def remove_duplicates(mat, mat_type):
    checker = set()
    new_mat = []
    if mat_type == 'alb' or mat_type == 'art':
        for person in mat:
            for entry in person:
                if entry[0] not in checker:
                    checker.add(entry[0])
                    new_mat.append(entry)
    elif mat_type == 'trk':
        for entry in mat:
            if entry[0] not in checker:
                checker.add(entry[0])
                new_mat.append(entry)

    return new_mat


song_features = remove_duplicates(all_song_features, 'trk')
print(knn_track_similarity_edges(np.array(song_features), 4))
