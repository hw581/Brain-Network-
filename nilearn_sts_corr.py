##Find Single subject maps of seed-to-seed correlation
##Compare with the Default Mode Network.
from nilearn.maskers import NiftiSpheresMasker
from nilearn import image
from nilearn.maskers import NiftiMasker
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import plot_connectome

# Sphere radius in mm
sphere_radius = 8
# Sphere center in MNI-coordinate
sphere_center = [ (1, 55, -3), #mPFC
                  (-50, -62, 36), (50, -62, 36),#ILPC
                  (1, -61, 38), #PCC
                  (25, -32, -18), (-25, -32, -18), #PHG
                  (59, -19, 10), (-59, 19, 10), #lateral temporal cortex
                  (24, 34, -12), (-24, 34, -12) #OFC
                  ]

data= image.load_img('filtered_func_data.nii.gz')

# Create masker object to extract average signal within spheres
masker = NiftiSpheresMasker(sphere_center, radius=sphere_radius, detrend=True,
standardize=True, verbose=1, memory="nilearn_cache", memory_level=2)

# Extract average signal in spheres with masker object
time_series = masker.fit_transform(data)

#plot the average signal per sphere
# fig = plt.figure(figsize=(8, 4))
# plt.plot(time_series)
# plt.xlabel('Scan number')
# plt.ylabel('Normalized signal')
# plt.tight_layout();
# plt.show()

connectivity_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrix = connectivity_measure.fit_transform([time_series])[0]
np.fill_diagonal(partial_correlation_matrix, 0)
# Plotting the partical correlation matrix
fig, ax = plt.subplots()
plt.imshow(partial_correlation_matrix, cmap='coolwarm')
for (j,i),label in np.ndenumerate(partial_correlation_matrix):
    ax.text(i, j, round(label, 2), ha='center', va='center', color='w')
    ax.text(i, j, round(label, 2), ha='center', va='center', color='w')
plt.colorbar()
plt.show()

# Plot the correlation matrix with a color bar
# plt.imshow(partial_correlation_matrix, cmap='coolwarm')
# plt.colorbar()
# plt.show()

threshold = 0.5
binary_matrix = partial_correlation_matrix > threshold
binarized_matrix = binary_matrix.astype(np.int_)
# for (j,i),label in np.ndenumerate(partial_correlation_matrix):
#     ax.text(i, j, round(label, 2), ha='center', va='center', color='w')
#     ax.text(i, j, round(label, 2), ha='center', va='center', color='w')
# plt.show()
plt.imshow(binary_matrix, cmap='coolwarm')
plt.colorbar()
plt.show()

plot_connectome(partial_correlation_matrix, sphere_center, display_mode='ortho', colorbar=True,  node_size=150, title="Default Mode Network Connectivity")
plotting.show()
