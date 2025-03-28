# Brain Network Analysis
These are codes for my summer research projects.
Codes for brain network analysis (determining dynamic brain networks from resting state fMRI data) are in the branch "master"

The results are in the pdf "Results.pdf"

Details are below:
# Set up:

We have preprocessed fMRI data (consisting of a set of time series corresponding to 64 times 64 times 33 voxels) and a list of change points. 

The time series is the signal measured at each 64 times 64 times 33 voxels, which correspond to locations in the brain. The change points for a subject are the times when the mean time series calculated from the whole brain of the subject experienced a change.

# Aim:

We want to
1) Create brain functional connectivity networks corresponding to time segments segmented by change points
2) Represent the brain networks as points in a latent space
3) Do clustering on these points
4) Compare brain networks with the Default Mode Network structure

# Codes:

1) GAE_data.py

This program outputs adjacency matrices (indicating connectivity networks) and correlation matrices as text files.

We first define ROIs (regions of interests) in the brain. They correspond to nodes in brain connectivity networks.

Here, msdl atlas is used to define ROIs. The msdl atlas has 39 ROIs.

Nodes in a brain connectivity network are connected if the correlation of time serieses for corresponding brain regions is over 1/2.

2) GAE500.py

This program outputs representations of input networks in latent space.

Graph Autoencoders are used.

3) kmeans.py

This program outputs clustering labels and properties of clusters (e.g. properties of networks in each cluster).

K-means clustering is used.

4) nilearn_stv_corr.py

This program allows us to check if the input network has the Default Mode Network structure.

The output shows the correlations between PCC (posterior cingulate cortex) and the regions in the whole brain (seed-to-voxel correlation). One of the typical features of the default mode network is the strong correlation between PCC and mPFC (medial prefrontal cortex).

5) nilearn_sts_corr.py

This program allows us to check if the input network has the Default Mode Network structure.

The output shows the correlations between 8 regions (e.g. PCC, mPFC, OFC) related to Default Mode Network (seed-to-seed correlation). One of the typical features of Deault Mode Network is the correlations between these areas.
