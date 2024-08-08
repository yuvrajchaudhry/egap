import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
def generate_coordinates(x_shape, kernel, stride, padding):
    assert len(x_shape) == 4
    assert len(kernel.shape) == 4
    assert x_shape[1] == kernel.shape[1]
    k_i, k_j = kernel.shape[-2:]
    x_i, x_j = np.array(x_shape[-2:])+2*padding
    y_i, y_j = (x_i-k_i)//stride+1, (x_j-k_j)//stride+1
    kernel = kernel.reshape(kernel.shape[0], -1)
    circulant_w = []
    for f in range(len(kernel)):
        circulant_row = []
        for u in range(len(kernel[f])):
            c = u // (k_i*k_j)
            h = (u - c*k_i*k_j) // k_j
            w = u - c*k_i*k_j - h*k_j
            rows = np.array(range(0, x_i-k_i+1, stride)) + h
            cols = np.array(range(0, x_j-k_j+1, stride)) + w
            circulant_unit = []
            for row in range(len(rows)):
                for col in range(len(cols)):
                    circulant_unit.append([f*y_i*y_j+row*y_j+col, c*x_i*x_j+rows[row]*x_j+cols[col]])
            circulant_row.append(circulant_unit)
        circulant_w.append(circulant_row)
    return np.array(circulant_w), x_shape[1]*x_i*x_j, kernel.shape[0]*y_i*y_j


def circulant_w(x_len, kernel, coors, y_len):
    weights = np.zeros([y_len, x_len], dtype=np.float32)
    kernel = kernel.reshape(kernel.shape[0], -1)
    for coor, f in list(zip(coors, kernel)):
        for c, v in list(zip(coor, f)):
            for h, w in c:
                assert weights[h, w] == 0
                weights[h, w] = v
    return weights


def aggregate_g(k, x_len, coors):
    k = k.squeeze()
    A_mat = []
    for coor in coors:
        A_row = []
        for c in coor:
            A_unit = np.zeros(shape=x_len, dtype=np.float32)
            for i in c:
                assert A_unit[i[1]] == 0
                A_unit[i[1]] = k[i[0]]
            A_row.append(A_unit)
        A_mat.append(A_row)
    A_mat = np.array(A_mat)
    return A_mat.reshape(-1, A_mat.shape[-1])



#DE Implementation
# def aggregate_g(k, x_len, coors, max_params=50000):
#     k = k.squeeze()
#
#     def fitness_function(params):
#         A_mat = []
#         idx = 0
#         for coor in coors:
#             A_row = []
#             for c in coor:
#                 A_unit = np.zeros(shape=x_len, dtype=np.float32)
#                 for i in c:
#                    # assert A_unit[i[1]] == 0
#                     A_unit[i[1]] = params[idx]
#                     idx += 1
#                 A_row.append(A_unit)
#             A_mat.append(A_row)
#         A_mat = np.array(A_mat)
#
#         # Calculate Mean Squared Error (MSE)
#         mse = np.mean((A_mat.reshape(-1, A_mat.shape[-1]) - k) ** 2)
#         return mse
#
#     # Determine the number of parameters needed
#     num_params = sum(len(c) for coor in coors for c in coor)
#     print(f'Number of Parameters: {num_params}')
#
#     # bounds = [(k.min(), k.max()) for _ in range(len(k) * x_len)]
#     bounds = [(k.min(), k.max())] * num_params
#
#     # Ensure that the problem size is not excessively large
#     if num_params > max_params:  # Example threshold for problem size
#         raise ValueError(f"Too many parameters to optimize ({num_params}). Check the setup.")
#
#     result = differential_evolution(fitness_function, bounds, strategy='best1bin', maxiter=100 , popsize=5, tol=0.01)
#     optimal_params = result.x
#
#     # Construct A_mat using optimal parameters
#     A_mat = []
#     idx = 0
#     for coor in coors:
#         A_row = []
#         for c in coor:
#             A_unit = np.zeros(shape=x_len, dtype=np.float32)
#             for i in c:
#                 A_unit[i[1]] = optimal_params[idx]
#                 idx += 1
#             A_row.append(A_unit)
#         A_mat.append(A_row)
#     A_mat = np.array(A_mat)
#
#     return A_mat.reshape(-1, A_mat.shape[-1])

#LBFGS Implementation
# def aggregate_g(k, x_len, coors, max_params=50000):
#     k = k.squeeze()
#
#     def fitness_function(params):
#         A_mat = []
#         idx = 0
#         for coor in coors:
#             A_row = []
#             for c in coor:
#                 A_unit = np.zeros(shape=x_len, dtype=np.float32)
#                 for i in c:
#                     # assert A_unit[i[1]] == 0
#                     A_unit[i[1]] = params[idx]
#                     idx += 1
#                 A_row.append(A_unit)
#             A_mat.append(A_row)
#         A_mat = np.array(A_mat)
#
#         # Calculate Mean Squared Error (MSE)
#         mse = np.mean((A_mat.reshape(-1, A_mat.shape[-1]) - k) ** 2)
#         return mse
#
#     # Determine the number of parameters needed
#     num_params = sum(len(c) for coor in coors for c in coor)
#     print(f'Number of Parameters: {num_params}')
#
#     # Ensure that the problem size is not excessively large
#     if num_params > max_params:  # Example threshold for problem size
#         raise ValueError(f"Too many parameters to optimize ({num_params}). Check the setup.")
#
#     # Set bounds for L-BFGS-B
#     bounds = [(k.min(), k.max())] * num_params
#
#     # Initial guess for the parameters (zero initialization)
#     initial_guess = np.zeros(num_params, dtype=np.float32)
#
#     # Run L-BFGS-B optimization
#     result = minimize(fitness_function, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'disp': True})
#
#     # Extract optimal parameters
#     optimal_params = result.x
#
#     # Construct A_mat using optimal parameters
#     A_mat = []
#     idx = 0
#     for coor in coors:
#         A_row = []
#         for c in coor:
#             A_unit = np.zeros(shape=x_len, dtype=np.float32)
#             for i in c:
#                 A_unit[i[1]] = optimal_params[idx]
#                 idx += 1
#             A_row.append(A_unit)
#         A_mat.append(A_row)
#     A_mat = np.array(A_mat)
#
#     return A_mat.reshape(-1, A_mat.shape[-1])

#Adam Implementation
# def aggregate_g(k, x_len, coors, num_iterations=100, lr=0.01, device='cuda'):
#     k = k.squeeze()
#
#     # Convert k to a PyTorch tensor and move to the desired device
#     k_tensor = torch.tensor(k, dtype=torch.float32, device=device)
#     print(f'k_tensor initialized with shape: {k_tensor.shape} on device: {device}')
#
#     def fitness_function(params, k_tensor):
#         A_mat = []
#         idx = 0
#         for coor in coors:
#             A_row = []
#             for c in coor:
#                 A_unit = torch.zeros(x_len, dtype=torch.float32, device=device)
#                 for i in c:
#                     A_unit[i[1]] = params[idx]  # Use tensor directly
#                     idx += 1
#                 A_row.append(A_unit)
#             A_mat.append(A_row)
#
#         A_mat = torch.stack([torch.stack(row) for row in A_mat])
#         print(f'A_mat shape: {A_mat.shape}')
#         print(f'k_tensor shape before reshape attempt: {k_tensor.shape}')
#
#         # Ensure k_tensor has the same shape as A_mat
#         if k_tensor.numel() != A_mat.numel():
#             raise ValueError(f"Cannot reshape k_tensor of shape {k_tensor.shape} with {k_tensor.numel()} elements to shape {A_mat.shape} with {A_mat.numel()} elements")
#
#         if k_tensor.shape != A_mat.shape:
#             print(f'Reshaping k_tensor from shape: {k_tensor.shape} to shape: {A_mat.shape}')
#             k_tensor = k_tensor.view(A_mat.shape)
#
#         # Calculate Mean Squared Error (MSE)
#         mse = torch.mean((A_mat - k_tensor) ** 2)
#         return mse
#
#     # Determine the number of parameters needed
#     num_params = sum(len(c) for coor in coors for c in coor)
#     print(f'Number of Parameters: {num_params}')
#
#     # Initialize parameters and move to the desired device
#     initial_params = torch.zeros(num_params, dtype=torch.float32, requires_grad=True, device=device)
#     print(f'initial_params initialized with shape: {initial_params.shape} on device: {device}')
#
#     # Create Adam optimizer
#     optimizer = torch.optim.Adam([initial_params], lr=lr)
#
#     # Optimization loop
#     for iteration in range(num_iterations):
#         optimizer.zero_grad()  # Clear previous gradients
#
#         # Calculate loss (fitness function)
#         loss = fitness_function(initial_params, k_tensor)  # Pass the current parameters and k_tensor to the fitness function
#
#         # Backpropagation
#         loss.backward()
#
#         # Update parameters
#         optimizer.step()
#
#         # Print loss every 10 iterations
#         if iteration % 10 == 0:
#             print(f"Iteration {iteration}, Loss: {loss.item()}")
#
#     # Convert final parameters to numpy array
#     optimal_params = initial_params.detach().cpu().numpy()
#
#     # Construct A_mat using optimal parameters
#     A_mat = []
#     idx = 0
#     for coor in coors:
#         A_row = []
#         for c in coor:
#             A_unit = np.zeros(x_len, dtype=np.float32)
#             for i in c:
#                 A_unit[i[1]] = optimal_params[idx]  # Use optimized parameters
#                 idx += 1
#             A_row.append(A_unit)
#         A_mat.append(A_row)
#     A_mat = np.array(A_mat)
#
#     return A_mat.reshape(-1, A_mat.shape[-1])
