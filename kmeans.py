##We have a set of points
##We want to implement K means clustering

##The program outputs clustering label for each point and their properties

import os
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import re
import math
from sklearn.metrics import pairwise_distances_argmin_min

##Set number of clusters
n_clus = 10

## Check that filenames match with subjects in data

#'latent500' stores a set of points
folder_path = 'latent500'
#make file list sorted by file name
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
# Define a regular expression to match the numerical part of the file name
pattern = re.compile(r'\d+')
# Define a function to extract the numerical part of the file name
def extract_number(filename):
    matches = pattern.findall(filename)
    if matches:
        return int(''.join(matches))
    else:
        return -1
# Sort the file names by increasing order of numbers
file_list = sorted(file_list, key=extract_number)
#file_list length is the number of subjects 
#Data list for all subjects
X = []
for i in range(len(file_list)):
  file_path = file_list[i]
  data = np.loadtxt(file_path)
  X.append(data)

X=np.array(X)

##Implement Kmeans clustering

kmeans = KMeans(n_clusters=n_clus,  n_init='auto', random_state=3).fit(X)

labels = kmeans.labels_

##Investigate properties

unique, counts = np.unique(labels, return_counts=True)
label_counts = dict(zip(unique, counts))
print(label_counts)

#labels for subjects with no changepoints (89 subjects had no changepoints)
labels_no_cp = labels[:90]

unique, counts = np.unique(labels_no_cp, return_counts=True)
label_no_cp_counts = dict(zip(unique, counts))
print(label_no_cp_counts)

#labels for subjects with changepoints (1st time segment)
labels_1 = labels[90:197]
#labels for subjects with changepoints (2nd time segment)
labels_2 = labels[197:304]
#labels for subjects with changepoints (3rd time segment)
labels_3 = labels[304:411]

#check if graphs from a subject are in the same cluster or not
diff_1_2 = labels_1 - labels_2
diff_3_2 = labels_3 - labels_2
diff_3_1 = labels_3 - labels_1

print("percentate of subjects time segment 1 and time segment 2 are in different cluster", np.count_nonzero(diff_1_2)/len(diff_1_2))
print("percentate of subjects time segment 3 and time segment 2 are in different cluster", np.count_nonzero(diff_3_2)/len(diff_3_2))
print("percentate of subjects time segment 1 and time segment 3 are in different cluster", np.count_nonzero(diff_3_1)/len(diff_3_1))



unique, counts = np.unique(labels_1, return_counts=True)
label_1_counts = dict(zip(unique, counts))
print(label_1_counts)

unique, counts = np.unique(labels_2, return_counts=True)
label_2_counts = dict(zip(unique, counts))
print(label_2_counts)

unique, counts = np.unique(labels_3, return_counts=True)
label_3_counts = dict(zip(unique, counts))
print(label_3_counts)

centroids = kmeans.cluster_centers_
print("cluster 0: ",  centroids[0])
print("cluster 1: ",  centroids[1])
print("cluster 5: ",  centroids[5])
print("cluster 7: ",  centroids[7])
print("cluster 9: ",  centroids[9])

dist59=math.dist(centroids[5], centroids[9])
dist01=math.dist(centroids[0], centroids[1])
dist05=math.dist(centroids[0], centroids[5])
dist09=math.dist(centroids[0], centroids[9])
dist15=math.dist(centroids[1], centroids[5])
dist07=math.dist(centroids[0], centroids[7])
dist17=math.dist(centroids[1], centroids[7])
dist57=math.dist(centroids[5], centroids[7])
dist79=math.dist(centroids[7], centroids[9])
dist19=math.dist(centroids[1], centroids[9])


print("dist01 : ", dist01)
print("dist05 : ", dist05)
print("dist09 : ", dist09)
print("dist15 : ", dist15)
print("dist19 : ", dist19)
print("dist59 : ", dist59)
print("dist07 : ", dist07)
print("dist17 : ", dist17)
print("dist57 : ", dist57)
print("dist79 : ", dist79)


check=np.zeros(107)
for i in range(107):
    if diff_1_2[i]==0:
        if labels_1[i]==labels_2[i]==labels_3[i]:
            check[i]=1

num_nonzeros = np.count_nonzero(check)          
print("all 3 time segments in the same cluster (number)",num_nonzeros)
print("all 3 time segments in the same cluster(location)",np.nonzero(check))

for i in np.nonzero(check):
    print(labels_1[i])

# Get the indices of the closest points to each centroid
closest, dist = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
print("closest points in each cluster", closest)
print("distance ", dist)

#cluster 1, 2, 8 = 0
#cluster 5, 6, 7 = 1
#cluster 0, 9 = 2
how_dense = labels
how_dense[(how_dense == 1) | (how_dense == 2)| (how_dense == 8)] = 1
how_dense[(how_dense == 5) | (how_dense == 6)| (how_dense == 7)] = 2
how_dense[(how_dense == 0) | (how_dense == 9)] = 3

#labels for subjects with changepoints (1st time segment)
dense_1 = how_dense[90:197]
#labels for subjects with changepoints (2nd time segment)
dense_2 = how_dense[197:304]
#labels for subjects with changepoints (3rd time segment)
dense_3 = how_dense[304:411]

dense_diff_1_2 = dense_1 - dense_2
dense_diff_3_2 = dense_3 - dense_2
dense_diff_3_1 = dense_3 - dense_1

print("percentate of subjects time segment 1 and time segment 2 are in different cluster", np.count_nonzero(dense_diff_1_2)/len(diff_1_2))
print("percentate of subjects time segment 3 and time segment 2 are in different cluster", np.count_nonzero(dense_diff_3_2)/len(diff_3_2))
print("percentate of subjects time segment 1 and time segment 3 are in different cluster", np.count_nonzero(dense_diff_3_1)/len(diff_3_1))

check=np.zeros(107)
for i in range(107):
    if dense_diff_1_2[i]==0:
        if dense_1[i]==dense_2[i]==dense_3[i]:
            check[i]=1

num_nonzeros = np.count_nonzero(check)          
print("all 3 time segments in the same dense clusters (number)",num_nonzeros)
print("all 3 time segments in the same dense clusters (location)",np.nonzero(check))
print(dense_2)
print(labels_2)

print(dense_1)
print(dense_3)