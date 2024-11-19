import tensorflow_datasets as tfds


dataset = tfds.load('d4rl_mujoco_halfcheetah/v2-expert', split='train')
#%%
import tensorflow as tf
import tensorflow_datasets as tfds

# Assuming 'dataset' is your PrefetchDataset
numpy_dataset = tfds.as_numpy(dataset)
dataset
#%%
for element in dataset:
    # Process each element
    print(element)