from utils import *
from conv2circulant import *
import torch
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os
from scipy.optimize import differential_evolution
from torch.optim import LBFGS
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import Adamax

setup = {'device': 'cpu', 'dtype': torch.float32}

def inverse_udldu(udldu, image_index=None):
    '''Derive u from udldu using Differential Evolution method.'''
    udldu = torch.tensor(udldu).to(**setup)

    os.makedirs("Plots", exist_ok=True)

    #Objective Function for DE
    def objective_func(u): #Logcosh
        u = torch.tensor(u).to(**setup)
        udldu_ = -u / (1 + torch.exp(u))
        loss = torch.mean(torch.log(torch.cosh(udldu_ - udldu))).item()
        return loss

    bounds = [(-7, 7)]  # Adjusted bounds
    popul = 30 # Population Size

    iteration_count = 0
    min_iterations = 100
    max_iterations = 1000
    convergence_iteration = None

    # Initialize the plot
    fig, ax = plt.subplots()
    # ax.set_xlim([-10, 10])
    # ax.set_ylim([-10, 10])  # Adjust as necessary for your specific problem
    # ax.set_xlabel('Solution Value (u)')
    # ax.set_ylabel('Objective Function Value (Loss)')
    # ax.set_title('Search Space of Solutions in Differential Evolution Optimization', fontsize=11)
    # ax.title.set_size(11)

    all_solutions = []
    all_objectives = []

    best_solution = float('inf')
    best_objective = float('inf')
    best_solution_value = float('inf')  # Track the best solution based on value
    best_solution_iteration = None

    def callback_function(xk, convergence):
        nonlocal iteration_count, convergence_iteration, best_solution, best_objective, best_solution_value,best_solution_iteration
        iteration_count += 1
        #print(f"Iteration: {iteration_count}")
        #current_objective = objective_quantile(xk)
        current_objective = objective_func(xk)
        print(f"Iteration: {iteration_count}, Current Solution: {xk[0]}, Objective Value: {current_objective}")

        all_solutions.append(xk[0])
        all_objectives.append(current_objective)

        if current_objective < best_objective:
            best_solution = xk
            best_objective = current_objective
            best_solution_iteration = iteration_count
            print(f"New best solution found: {best_solution}, with objective value: {best_objective} at iteration {best_solution_iteration}")


        # Update the best solution based on the current solution's absolute value
        # if abs(xk[0]) < best_solution_value:
        # elif current_objective == 0.0:
        #     # Even if the current objective is equal to the best (i.e., 0), keep checking for a smaller solution
        #     if abs(xk[0]) < best_solution_value:
        #         best_solution = xk
        #         best_solution_value = abs(xk[0])
        #         best_objective = current_objective  # Keep objective as 0
        #         best_solution_iteration = iteration_count
        #         print(f"New best solution found (objective = 0): {best_solution}, with objective value: {best_objective}, at iteration {best_solution_iteration}")

        ax.clear()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])  # Adjust as necessary for your specific problem
        ax.set_xlabel('Solution Value (u)')
        ax.set_ylabel('Objective Function Value (Loss)')
        ax.set_title('Search Space of Solutions', fontsize=11)
        ax.title.set_size(11)

        if all_solutions:
            ax.scatter(all_solutions, all_objectives, color='blue', label='All Solutions')
        if best_solution is not None:
            ax.scatter(best_solution[0], best_objective, color='red', label='Best Solution')
        ax.legend()
        plt.draw()
        #plt.pause(0.1)
        if convergence and convergence_iteration is None:
            convergence_iteration = iteration_count
            print(f"Convergence detected at iteration {iteration_count}")

        if iteration_count < min_iterations:
            return False  # Continue optimization

        # Stop condition
        if convergence or iteration_count >= max_iterations:
            if convergence:
                #print(f"Stopping optimization at iteration {iteration_count} as minimum iterations are completed. Convergence was already detected at {convergence_iteration}")
                print(f"Stopping optimization at iteration {iteration_count} as minimum iterations are completed and convergence is done. Best solution detected at iteration {best_solution_iteration}.")
            else:
                print(f"Maximum iterations exceeded {iteration_count}")
            return True  # Stop the optimization

        return False


    # Perform Differential Evolution optimization
    # result = differential_evolution(objective_quantile, bounds, popsize=10, callback=callback_function, maxiter=max_iterations, polish=False, init='latinhypercube', updating='deferred')
    while iteration_count < min_iterations:
        # result = differential_evolution(objective_quantile, bounds, popsize=10, callback=callback_function,
        #                                 maxiter=max_iterations, polish=False, init='latinhypercube',
        #                                 updating='deferred')
        result = differential_evolution(objective_func, bounds, popsize=popul, callback=callback_function,
                                        maxiter=max_iterations, polish=False, init='latinhypercube',
                                        updating='deferred')
        if result.success and iteration_count >= min_iterations:
            best_solution = result.x
            best_objective = result.fun
            break

    #u_optimized = result.x[0]
    u_optimized = best_solution[0]

    u = torch.tensor(u_optimized).to(**setup)
    udldu_ = -u / (1 + torch.exp(u))

    print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
    print(f"Best solution: {best_solution}")
    print(f"Best objective value: {best_objective}")
    print(f"Optimal solution: {u_optimized}")
    print(f"Bounds used: {bounds}")
    print(f"Population: {popul}")
    print(f"Total number of iterations performed: {iteration_count}")
    # print(f"Total number of iterations required: {result.nit}")
    print(f"Total number of iterations required: {best_solution_iteration}")

    if image_index is not None:
        plot_save_path = os.path.join("Plots", f"Img_{image_index}.png")
    else:
        plot_save_path = os.path.join("Plots", "Img.png")

    plt.savefig(plot_save_path)
    plt.close(fig)  # Close the figure to free memory

    print(f"Plot saved as: {plot_save_path}")

    return u_optimized

def peeling(in_shape, padding):
    if padding == 0:
        return np.ones(shape=in_shape, dtype=bool).squeeze()
    h, w = np.array(in_shape[-2:]) + 2*padding
    toremain = np.ones(h*w*in_shape[1], dtype=bool)
    if padding:
        for c in range(in_shape[1]):
            for row in range(h):
                for col in range(w):
                    if col < padding or w-col <= padding or row < padding or h-row <= padding:
                        i = c*h*w + row*w + col
                        assert toremain[i]
                        toremain[i] = False
    return toremain


def padding_constraints(in_shape, padding):
    toremain = peeling(in_shape, padding)
    P = []
    for i in range(toremain.size):
        if not toremain[i]:
            P_row = np.zeros(toremain.size, dtype=np.float32)
            P_row[i] = 1
            P.append(P_row)
    return np.array(P)


def cnn_reconstruction(in_shape, k, g, out, kernel, stride, padding):
    coors, x_len, y_len = generate_coordinates(x_shape=in_shape, kernel=kernel, stride=stride, padding=padding)
    K = aggregate_g(k=k, x_len=x_len, coors=coors)
    W = circulant_w(x_len=x_len, kernel=kernel, coors=coors, y_len=y_len)
    P = padding_constraints(in_shape=in_shape, padding=padding)
    p = np.zeros(shape=P.shape[0], dtype=np.float32)
    if np.any(P):
        a = np.concatenate((K, W, P), axis=0)
        b = np.concatenate((g.reshape(-1), out, p), axis=0)
    else:
        a = np.concatenate((K, W), axis=0)
        b = np.concatenate((g.reshape(-1), out), axis=0)
    result = np.linalg.lstsq(a, b, rcond=None)
    print(f'lstsq residual: {result[1]}, rank: {result[2]} -> {W.shape[-1]}, '
          f'max/min singular value: {result[3].max():.2e}/{result[3].min():.2e}')
    x = result[0]
    return x[peeling(in_shape=in_shape, padding=padding)], W


def fcn_reconstruction(k, gradient):
    x = [g / c for g, c in zip(gradient, k) if c != 0]
    x = np.mean(x, 0)
    return x


def r_gap(out, k, g, x_shape, weight, module):
    # obtain information of convolution kernel
    if isinstance(module.layer, nn.Conv2d):
        padding = module.layer.padding[0]
        stride = module.layer.stride[0]
    else:
        padding = 0
        stride = 1

    x, weight = cnn_reconstruction(in_shape=x_shape, k=k, g=g, kernel=weight, out=out, stride=stride, padding=padding)
    return x, weight

