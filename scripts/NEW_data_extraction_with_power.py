import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import kron, norm
from R_func import R_func
#from cvx_func_bisection import cvx_func_bisection_2
#from cvx_func_bisection_rate import cvx_func_bisection_2
#from cvx_mod import cvx_func_bisection_2
import pandas as pd
import pickle 
import pyarrow as pa
import pyarrow.parquet as pq
import os
import matlab.engine

np.random.seed(42)

# Initialize the DataFrame
file_path = '3_users_data.pkl'

# Check if the pickle file exists and is non-empty
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    # Load the existing DataFrame from the pickle file
    with open(file_path, 'rb') as f:  # Open in read-binary mode
        df = pickle.load(f)
else:
    print("File is empty or does not exist, initializing a new DataFrame.")
    df = pd.DataFrame(columns=['H', 'power', 'label'])



def call_cvx_in_matlab(eng, t, M, K, sigma, p, H):
    # Convert to MATLAB-compatible column vectors
    sigma_col = sigma.reshape(-1, 1)  # Reshape to column
    p_col = p.reshape(-1, 1)          # Reshape to column
    H_mat = matlab.double(H.tolist())

    # Pass to MATLAB
    W, d, status = eng.cvx_func_bisection_2(
        matlab.double([t]), 
        matlab.double([M]), 
        matlab.double([K]), 
        matlab.double(sigma_col.tolist()), 
        matlab.double(p_col.tolist()), 
        H_mat, 
        nargout=3
    )
    return np.array(W), np.array(d), status

# Start MATLAB engine and add paths
eng = matlab.engine.start_matlab()
eng.addpath('cvx_func_bisection_2.m', nargout=0)  # Path to 
eng.addpath(eng.genpath(r'C:\Users\ayoubsa\Desktop\cvx'), nargout=0)  # Path to CVX toolbox



# Implementation of the main code
K = 3  # number of UE
M = 6  # number of transmitter
P_n_dB = np.arange(15, 31, 1)  # Power values in dBm
Pn = 10 ** ((P_n_dB - 30) / 10)  # Power values in linear scale
theta_c_k = 60 * (np.pi / 180)  # Receiver field of view in radians (60 deg -> rad)
q = 1.5  # Refractive index of optical concentrator
B = 1e8  # Bandwidth in Hz
BER = 1e-3  # Bit Error Rate
ee = 1.60217662e-19  # Elementary charge
A_PDk = 1e-4  # Photo Detector area
xi = 10.93  # Ambient light photocurrent
rho_k = 0.4  # Photo Detector responsivity
i_amp = 5e-12  # Preamplifier noise density
m = 1  # mode number of Lambertian emission
A_k = q ** 2 * A_PDk / (np.sin(theta_c_k)) ** 2  # Collection area

#9a3ed ygeneri fi des matrices mta3 les pos mta3 el users ely houma 4 w 9a3ed ygeneri bel 50 matrices
def generate_unique_matrices(num_matrices, num_rows, num_columns, mean, std_dev, room_dimensions):
    """
    Generate unique matrices of user positions within a specified 3D room.

    Parameters:
    - num_matrices: Number of unique matrices to generate.
    - num_rows: Number of users (rows per matrix).
    - num_columns: Number of dimensions (should be 3 for x, y, z).
    - mean: Mean value for the Gaussian distribution.
    - std_dev: Standard deviation for the Gaussian distribution.
    - room_dimensions: Tuple (x_max, y_max, z_max) defining room boundaries.

    Returns:
    - List of unique matrices with user positions.
    """
    x_max, y_max, z_max = room_dimensions
    matrices = []
    
    while len(matrices) < num_matrices:
        # Generate a matrix of random values distributed according to a Gaussian distribution
        matrix = np.random.normal(loc=mean, scale=std_dev, size=(num_rows, num_columns))
        
        # Clip the values to ensure they are within room dimensions
        matrix[:, 0] = np.clip(matrix[:, 0], 0, x_max)  # x-coordinate
        matrix[:, 1] = np.clip(matrix[:, 1], 0, y_max)  # y-coordinate
        matrix[:, 2] = np.clip(matrix[:, 2], 0, z_max)  # z-coordinate
        
        # Check if the generated matrix is unique
        if all((matrix != existing).any() for existing in matrices):
            matrices.append(matrix)
    
    return matrices

#transmitters position
t = np.array([[1, 1, 0],
              [4, 1, 0],
              [1, 2.5, 0],
              [4, 2.5, 0],
              [1, 4, 0],
              [4, 4, 0]])

#Users position
# Mean and standard deviation of the values 
# 4800 0.7 
# 80 0.1 
# 160 0.1 
# 480 0.7 
# 10560 9ad9adb 
# 2576 0.7 
# 2576 0.1 
# 15712 9ad9ad


#matlab_cvx 800*0.7 1600*0.1 1600*0.7 1600*0.1 1600*0.7 800*0.1 800*0.7 800*0.1  tot 9600
# 800*0.1 800*0.7 800*0.1 800*0.7 800*0.1 1600*0.7 800*0.1 tot 1600 
# 1600*0.1 1600*0.7 800*0.1 800*0.7 800*0.1 800*0.7 800*0.1 800*0.7 24000 
# 800*0.7 800*0.1 800*0.7 800*0.1 800*0.7 800*0.1 800*0.7 800*0.1 800*0.7 800*0.1 32000 
# 800*0.1 800*0.7 800*0.1 800*0.7 800*0.1 1600*0.7 800*0.1 800*0.7


#matlab_cvx 800*0.7 1600*0.1 1600*0.7 1600*0.1 1600*0.7 800*0.1 800*0.7 800*0.1 
#1600*0.1 1600*0.7 800*0.1 800*0.7 800*0.1



#data 20_users
#5008 *0.7 5008 *0.1 5008 *0.7

#data 6_users
#4800 *0.1 4800 *0.7 4800 *0.1 4800 *0.7 4800 *0.1 4800 *0.7 4800 *0.1 4800 *0.7

#data 2 users
#1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1

#1600 *0.7 1600 *0.1 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 

#data 3-2 users
#1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1

#data 3 users
#1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7

#data 5 users
#1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1

#data 1 users
#1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7 1600 *0.1 1600 *0.7  3200 *0.1 3200 *0.7
mean_u = 2.15
std_dev_u= 0.7


num_matrices_u = 100

num_rows_u = 3
num_columns_u = 3
a=5
room_dims = (5, 5, 3) # Room dimensions: 5x5x3

# Generate matrices
# Generate unique matrices for users
#u_matrices = generate_unique_matrices(num_matrices_u, num_rows_u, num_columns_u, mean_u, std_dev_u, a)
# Generate unique user position matrices
u_matrices = generate_unique_matrices(num_matrices_u, num_rows_u, num_columns_u, mean_u, std_dev_u, room_dims)


# Define the loop for generating data
for l, u in enumerate(u_matrices):
    # Initialize channel matrix H with zeros to store channel information
    H = np.zeros((M, K))
    # Calculate channel matrix H
    for i in range(M):  # Loop over transmitters
        for j in range(K):  # Loop over users
            d1 = np.linalg.norm(u[j, :] - t[i, :])  # Distance between user and transmitter
            tt = t[i, :].copy()
            tt[2] = u[j, 2]
            d2 = np.linalg.norm(u[j, :] - tt)  # Updated distance for angle calculation
            phi = np.arcsin(d2 / d1)
            theta = np.arccos(d2 / d1)
            if theta <= theta_c_k:
                H[i, j] = rho_k * A_k / d1 ** 2 * R_func(phi, m) * np.cos(theta)

    # Prepare to store data for each Pn[n]
    for n in range(len(Pn)):
        p = Pn[n] * np.ones(M)
        Ps = Pn[n] * np.sum(H, axis=0)  # Total power received at each user
        sigma = np.zeros(K)

        for i in range(K):  # Loop over users
            sigma[i] = (
                2 * ee * Ps[i] * B
                + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B
                + i_amp ** 2 * B
            )

        W = H
        x = np.zeros((M, 1))
        for ll in range(M):
            x[ll] = np.linalg.norm(W[ll, :], ord=1)
        val = np.max(x)
        W = W / val * Pn[n]

        it = 0
        t_aux = 1e-6
        t_works = 1e-10
        
        while abs(t_works - t_aux) / t_works >= 1e-3 or it <= 1:
            it += 1
            t_aux = t_works
            t_lower = t_works
            t_upper = 1e5
            tol = 0.01
            while (t_upper - t_lower) / t_lower > tol:
                t_test = (t_upper + t_lower) / 2
                Wz, d, status = call_cvx_in_matlab(eng,t_test, M, K, sigma, p, H)
                if status != "Solved":
                    t_upper = t_test
                else:
                    W_works = Wz
                    t_works = t_test
                    t_lower = t_test
            W = W_works
        #l.append(W)
        # Store data for this H, Pn[n], and W
        df = df._append({'H': H, 'power': Pn[n], 'label': W}, ignore_index=True)

# Save the updated DataFrame to the pickle file
with open('3_users_data.pkl', 'wb') as f:
    pickle.dump(df, f)

print(f"Data saved with shape: {df.shape}")
