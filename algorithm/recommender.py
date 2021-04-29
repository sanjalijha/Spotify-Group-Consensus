import numpy as np
import scipy
import sklearn
from sklearn import metrics
from demo_data import data, album_track_data, artist_albums_data, all_song_features
import sys

np.set_printoptions(threshold=sys.maxsize)

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
no_of_vertices = 0

no_of_e3 = 0
no_of_e7 = 0
no_of_e8 = 0
no_of_e9 = 0
no_of_edges = 0


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

e7 = remove_duplicates(album_track_data, 'alb')
no_of_e7 = len(e7)

e8 = remove_duplicates(artist_albums_data,'art')
no_of_e8 = len(e8)


song_features = remove_duplicates(all_song_features, 'trk')
e9 = knn_track_similarity_edges(np.array(song_features), 4)
no_of_e9 = len(e9)

no_of_edges = no_of_e9 + no_of_e8 + no_of_e7 + no_of_e3
no_of_vertices = no_of_users + no_of_tracks + no_of_albums + no_of_artists

H = np.zeros((no_of_vertices,no_of_edges))
print(H.shape)

for i in range(no_of_e3):
  user = int(e3[i][0])
  track_id = e3[i][1]
  track_index = tracks_indices[track_id]
  H[user,i] = 1
  H[track_index,i] = 1

for i in range(no_of_e7):
  ind = no_of_e3 + i
  edge = e7[i]
  for j in range(len(edge)):
    if j == 0:
      row = albums_indices[edge[j]]
      H[row,ind] = 1
    else:
      if edge[j] in tracks_indices.keys():
        row = tracks_indices[edge[j]]
        H[row,ind] = 1

for i in range(no_of_e8):
  ind = no_of_e3 + no_of_e7 + i
  edge = e8[i]
  for j in range(len(edge)):
    if j == 0:
      row = artists_indices[edge[j]]
      H[row,ind] = 1
    else:
      if edge[j] in albums_indices.keys():
        row = albums_indices[edge[j]]
        H[row,ind] = 1

for i in range(no_of_e9):
  ind = no_of_e3 + no_of_e7 + no_of_e8 + i
  track_1 = tracks_indices[e9[i][0]]
  track_2 = tracks_indices[e9[i][1]]
  H[track_1,ind] = 1
  H[track_2,ind] = 1

De_arr = np.sum(H,axis=0)
De = np.zeros((no_of_edges,no_of_edges))
for i in range(no_of_edges):
  De[i,i] = 1/De_arr[i]

W = np.zeros((no_of_edges, no_of_edges))
limit = 50
weight = 50
norm = 25
for e in range(limit):
    W[e, e] = weight/norm
    weight -= 1
weight = 50
for e in range(limit, 2*limit):
    W[e, e] = weight/norm
    weight -= 1
weight = 50
for e in range(2*limit, 3*limit):
    W[e, e] = weight/norm
    weight -= 1
weight = 50
for e in range(3*limit, 4*limit):
    W[e, e] = weight/norm
    weight -= 1
for e in range(4*limit, 8*limit):
    W[e, e] = 1/norm
for e in range(8*limit, 8*limit+no_of_e7+no_of_e8):
    W[e, e] = 1
s = 0
for e, i in zip(range(8*limit+no_of_e7+no_of_e8, no_of_edges), range(no_of_e9)):
    W[e, e] = e9[i][2]

Dv = np.zeros((no_of_vertices,no_of_vertices))

for i in range(no_of_vertices):
  total = 0
  for j in range(no_of_edges):
    total += H[i,j]*W[j,j]
  if total != 0:
    Dv[i,i] = 1/(total**0.5)
  else:
    Dv[i,i] = 0

A = Dv@H@W@De@H.T@Dv
print(A.shape)

y = np.zeros(no_of_vertices)
y[:4] = 1

f = np.linalg.inv((np.identity(no_of_vertices) - (2/3)*A))@y
# print(f)

indices = np.argsort(f*-1)

songs = []
idx = 0

for i in range(len(indices)):
  for key, value in tracks_indices.items():
    if value == indices[i]:
      songs.append(key)
  
print(songs[0:20])
