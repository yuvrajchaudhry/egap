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
#
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

# def generate_coordinates(x_shape, kernel, stride, padding):
#     assert len(x_shape) == 4, "x_shape must be a 4D tensor"
#     assert len(kernel.shape) == 4, "Kernel must be a 4D tensor"
#     assert x_shape[1] == kernel.shape[1], "Channel dimensions must match"
#     k_i, k_j = kernel.shape[-2:]  # Kernel height and width
#     x_i, x_j = np.array(x_shape[-2:]) + 2 * padding  # Input dimensions with padding
#     y_i, y_j = (x_i - k_i) // stride + 1, (x_j - k_j) // stride + 1  # Output dimensions
#     kernel_flat = kernel.reshape(kernel.shape[0], -1)  # Flatten the kernel
#     circulant_w = []  # List to store circulant coordinates
#     for f in range(len(kernel_flat)):
#         circulant_row = []
#         for u in range(len(kernel_flat[f])):
#             c = u // (k_i * k_j)  # Channel index
#             h = (u % (k_i * k_j)) // k_j  # Row index in kernel
#             w = u % k_j  # Column index in kernel
#             # Compute valid row and column positions
#             rows = np.arange(0, x_i - k_i + 1, stride) + h
#             cols = np.arange(0, x_j - k_j + 1, stride) + w
#             # Generate circulant unit coordinates
#             circulant_unit = [
#                 [f * y_i * y_j + row * y_j + col, c * x_i * x_j + rows[row] * x_j + cols[col]]
#                 for row in range(len(rows)) for col in range(len(cols))
#             ]
#             circulant_row.append(circulant_unit)
#         circulant_w.append(circulant_row)
#     return np.array(circulant_w), x_shape[1] * x_i * x_j, kernel_flat.shape[0] * y_i * y_j


# def circulant_w(x_len, kernel, coors, y_len):
#     weights = np.zeros((y_len, x_len), dtype=np.float32)  # Initialize weights matrix
#     kernel_flat = kernel.reshape(kernel.shape[0], -1)  # Flatten kernel for iteration
#     for coor, f in zip(coors, kernel_flat):
#         for c, v in zip(coor, f):
#             for h, w in c:
#                 if weights[h, w] != 0:
#                     raise ValueError(f"Weight at ({h}, {w}) is already set. Conflict with value: {weights[h, w]}")
#                 weights[h, w] = v  # Assign value to weights matrix
#     return weights


# def aggregate_g(k, x_len, coors):
#     k = k.squeeze()  # Ensure k is a 1D array
#     A_mat = []  # List to store aggregated matrices
#     for coor in coors:
#         A_row = []
#         for c in coor:
#             A_unit = np.zeros(x_len, dtype=np.float32)  # Initialize unit array
#             for i in c:
#                 if A_unit[i[1]] != 0:
#                     raise ValueError(f"Position {i[1]} in A_unit is already set. Conflict with value: {A_unit[i[1]]}")
#                 A_unit[i[1]] = k[i[0]]  # Assign value from k to A_unit
#             A_row.append(A_unit)
#         A_mat.append(A_row)
#     A_mat = np.array(A_mat)  # Convert list of lists to NumPy array
#     return A_mat.reshape(-1, A_mat.shape[-1])  # Reshape to desired output format


