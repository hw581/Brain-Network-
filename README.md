Setup:

We have preprocessed fMRI data (time series corresponding to 64 times 64 times 33 voxels) and a list of changepoints in time

We want to
1)create brain networks corresponding to time segment segmented by changepoints
2)represent them as points in a latent space
3)do clustering these on points
4)compare data with Default Mode Network

Code:

1) GAE_data.py outputs adjacency matrices (indicating connectivity networks) and correlation matrices as text files

##We firstly define region of interests in brain (corresponds to node in brain networks)

##msdl atlas(39ROIs) is used to define region of interests (ROIs)

##Nodes are connected if the correlation of corresponding brian regions is over 1/2

2) GAE500.py outputs representations of input networks in latent space

##Use Graph Autoencoders

3) kmeans.py output clustring labels and properties of clusters

##Use K-means clustering
4) nilearn_sts_corr.py compares with DMN
