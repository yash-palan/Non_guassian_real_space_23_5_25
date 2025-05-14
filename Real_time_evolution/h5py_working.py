import h5py
import numpy as np


filename = "trial.h5"

# Function to create an HDF5 file and initialize a dataset
def create_hdf5_file(filename):
    with h5py.File(filename, "w") as f:
        # Create a dataset with maxshape=(None,) so it can grow
        f.create_dataset("time_series", shape=(0,0), maxshape=(None,None), dtype="f8")
        print("HDF5 file initialized.")

# Function to append data to the existing dataset
def append_to_hdf5(new_data,filename):
    with h5py.File(filename, "a") as f:
        dset = f["time_series"]
        old_size = dset.shape[0]  # Get the current size
        if(dset.shape[0] == 0):
            dset.resize((1,len(new_data)))
            dset[0,:] = new_data  # Append new data
            print(f"Appended {len(new_data)} entries to the dataset.") 
        else:
            second_size = dset.shape[1]  # Get the current size
            new_size = old_size + 1  # Calculate new size
            dset.resize((new_size,second_size))  # Resize the dataset
            dset[new_size-1,:] = new_data  # Append new data
            print(f"Appended {len(new_data)} entries to the dataset.")

# Example Usage
if __name__ == "__main__":
    create_hdf5_file()  # Run this only once to initialize
    for t in range(5):  # Simulate time steps
        new_data = np.random.rand(10)  # Generate some data (replace with your simulation results)
        append_to_hdf5(new_data)