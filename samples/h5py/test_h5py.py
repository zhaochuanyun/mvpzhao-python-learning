import h5py
import numpy as np

X = np.random.rand(3, 2).astype('float32')
y = np.random.rand(1, 2).astype('float32')

print('write X : ' + str(X))
print('write y : ' + str(y))

# Create a new file
f = h5py.File('data.h5', 'w')
f.create_dataset('X_train', data=X)
f.create_dataset('y_train', data=y)
f.close()

# Load hdf5 dataset
f = h5py.File('data.h5', 'r')
print(list(f.keys()))
X = np.array(f['X_train'][:])
Y = np.array(f['y_train'][:])

print('read X : ' + str(X))
print('read y : ' + str(y))
f.close()
