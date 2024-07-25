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


# def aggregate_g(k, x_len, coors):
#     k = k.squeeze()
#     A_mat = []
#     for coor in coors:
#         A_row = []
#         for c in coor:
#             A_unit = np.zeros(shape=x_len, dtype=np.float32)
#             for i in c:
#                 assert A_unit[i[1]] == 0
#                 A_unit[i[1]] = k[i[0]]
#             A_row.append(A_unit)
#         A_mat.append(A_row)
#     A_mat = np.array(A_mat)
#     return A_mat.reshape(-1, A_mat.shape[-1])


# def aggregate_g(k, x_len, coors, bounds, F=0.5, CR=0.9):
#   """
#   This function aggregates gradients using Differential Evolution (DE).
#
#   Args:
#       k: Gradients (numpy array, shape=(num_gradients,)).
#       x_len: Length of the final aggregated gradient (int).
#       bounds: Lower and upper bounds for the search space (list of tuples, [(min1, max1),...]).
#       F: Scaling factor for the differential mutation (float, default=0.5).
#       CR: Crossover probability (float, default=0.9).
#
#   Returns:
#       Aggregated gradient (numpy array, shape=(x_len,)).
#   """
#
#   # Population size (can be adjusted for better performance)
#   pop_size = 10
#
#   #Converting bounds to numpy array
#   bounds = np.array(bounds)
#
#   # Initialize population with random values within bounds
#   lower_bounds = bounds[:,0]
#   higher_bounds = bounds[:, 1]
#   population = np.random.uniform(low=lower_bounds, high=higher_bounds, size=(pop_size, x_len))
#
#   # Loop for a fixed number of iterations (can be adjusted)
#   for _ in coors:
#
#     # Generate mutant vector
#     for i in range(pop_size):
#       r1, r2, r3 = np.random.choice(pop_size, size=3, replace=False)
#       mutant = population[r1] + F * (population[r2] - population[r3])
#
#       # Clip mutant to stay within bounds
#       mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
#
#       # Crossover
#       trial_vector = np.copy(population[i])
#       crossover_mask = np.random.rand(x_len) < CR
#       trial_vector[crossover_mask] = mutant[crossover_mask]
#
#       # Evaluate fitness (replace with your actual gradient aggregation logic)
#       fitness_i = rgapfit(trial_vector)
#       fitness_orig = rgapfit(population[i])
#
#       # Selection
#       if fitness_i < fitness_orig:
#         population[i] = trial_vector
#
#   # Select best individual as the aggregated gradient
#   best_index = np.argmin([rgapfit(v) for v in population])
#   return population[best_index]
#
# # Replace this with your actual function that calculates the fitness based
# # on your gradient aggregation logic (e.g., loss on the model)
# def rgapfit(gradient):
#   nn.MSELoss(gradient)
#   pass


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

def aggregate_g(k, x_len, coors, num_iterations=100, lr=0.01):
    k = k.squeeze()

    # Convert k to a PyTorch tensor
    k_tensor = torch.tensor(k, dtype=torch.float32)  # Make sure this is done at the start

    def fitness_function(params):
        A_mat = []
        idx = 0
        for coor in coors:
            A_row = []
            for c in coor:
                A_unit = np.zeros(shape=x_len, dtype=np.float32)
                for i in c:
                    A_unit[i[1]] = params[idx].item()  # Extract scalar value from tensor
                    idx += 1
                A_row.append(A_unit)
            A_mat.append(A_row)
        A_mat = np.array(A_mat)

        # Convert A_mat to a PyTorch tensor
        A_mat_tensor = torch.tensor(A_mat.reshape(-1, x_len), dtype=torch.float32)

        # Ensure k_tensor has the same shape as A_mat_tensor
        if k_tensor.shape != A_mat_tensor.shape:
            k_tensor = k_tensor.view(A_mat_tensor.shape)

        # Calculate Mean Squared Error (MSE)
        mse = torch.mean((A_mat_tensor - k_tensor) ** 2)
        return mse

    # Determine the number of parameters needed
    num_params = sum(len(c) for coor in coors for c in coor)
    print(f'Number of Parameters: {num_params}')

    # Initialize parameters
    initial_params = torch.zeros(num_params, dtype=torch.float32, requires_grad=True)

    # Create Adam optimizer
    optimizer = torch.optim.Adam([initial_params], lr=lr)

    # Optimization loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients

        # Calculate loss (fitness function)
        loss = fitness_function(initial_params)  # Pass the current parameters to the fitness function

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print loss every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

    # Convert final parameters to numpy array
    optimal_params = initial_params.detach().numpy()

    # Construct A_mat using optimal parameters
    A_mat = []
    idx = 0
    for coor in coors:
        A_row = []
        for c in coor:
            A_unit = np.zeros(shape=x_len, dtype=np.float32)
            for i in c:
                A_unit[i[1]] = optimal_params[idx]  # Use optimized parameters
                idx += 1
            A_row.append(A_unit)
        A_mat.append(A_row)
    A_mat = np.array(A_mat)

    return A_mat.reshape(-1, A_mat.shape[-1])