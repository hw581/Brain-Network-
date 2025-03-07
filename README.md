# Set up:

We have preprocessed fMRI data (consisting of a set of time seriese corresponding to 64 times 64 times 33 voxels) and a list of changepoints. 

The time series is the signal measured at each 64 times 64 times 33 voxels which correspond to locations in the brain. The changepoints for a subjsct are the times when the mean time series calculated from the whole brain of the subject experienced a change.

# Aim:

We want to
1) create brain functional connectivity networks corresponding to time segment segmented by changepoints
2) represent the brain networks as points in a latent space
3) do clustering on these points
4) compare brain networks with the Default Mode Network structure

# Codes:

1) GAE_data.py

This program outputs adjacency matrices (indicating connectivity networks) and correlation matrices as text files.

We firstly define ROIs (regions of interests) in brain. They correspond to nodes in brain connectivity networks.

Here, msdl atlas is used to define ROIs. The msdl atlas has 39 ROIs.

Nodes in a brain connectivity network are connected if the correlation of time serieses for corresponding brian regions is over 1/2.

2) GAE500.py

This program outputs representations of input networks in latent space.

Graph Autoencoders is used.

3) kmeans.py

This program outputs clustring labels and properties of clusters (e.g. properties of networks in each cluster).

K-means clustering is used.

4) nilearn_stv_corr.py

This program allows us to check if the input network has the Default Mode Network structure.

The output shows the correlations between PCC (posterior cingulate cortex) and the regions in whole the brain (seed-to-voxel correlation). One of the typical features of Deault Mode Network is the strong correlation between PCC and mPFC (medial prefrontal cortex).

5) nilearn_sts_corr.py

This program allows us to check if the input network has the Default Mode Network structure.

The output shows the correlations between 8 regions (e.g. PCC, mPFC, OFC) related to Default Mode Network (seed-to-seed correlation). One of the typical features of Deault Mode Network is the correlations between these areas.
