# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:23:14 2024

@author: armin
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'dataset' is already defined
flatten_states = [step['action'].tolist() for episode in dataset for step in episode]
X = np.array(flatten_states)
#%%
# Apply PCA with 3 components
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(X)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the reduced data
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c='r', marker='o')

# Label axes
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

# Show the plot
plt.show()
#%%
import plotly.express as px
flatten_states = [step['action'].tolist() for episode in dataset for step in episode]
X = np.array(flatten_states)

# Apply PCA with 3 components
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(X)

# Create a DataFrame for plotly
df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2', 'PCA3'])

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3', title='PCA Reduced Data')

# Show the plot
fig.write_html("pca_3d_plot.html")
fig.write_image("pca_3d_plot.png")
#%%

df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])

# Plot the density heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.kdeplot(x='Component 1', y='Component 2', data=df, cmap="viridis", fill=True)
plt.title('Density Heatmap of PCA Reduction to 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()