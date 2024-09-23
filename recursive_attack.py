from utils import *
from conv2circulant import *
import torch
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from torch.optim import LBFGS
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import Adamax

setup = {'device': 'cpu', 'dtype': torch.float32}


def inverse_udldu(udldu):
    '''Derive u from udldu using Differential Evolution method and plot the optimization process.'''
    udldu = torch.tensor(udldu).to(**setup)

    def objective_quantile(u, quantile=0.5):
        u = torch.tensor(u).to(**setup)
        udldu_ = -u / (1 + torch.exp(u))
        loss = torch.mean(torch.max((quantile - (udldu_ - udldu) < 0) * (quantile - (udldu_ - udldu)),
                                    (1 - quantile - (udldu_ - udldu) > 0) * (1 - quantile - (udldu_ - udldu)))).item()
        return loss

    # def objective_mse(u):
    #     u = torch.tensor(u).to(**setup)
    #     udldu_ = -u / (1 + torch.exp(u))
    #     loss = torch.mean((udldu_ - udldu) ** 2).item()
    #     return loss

    # Define bounds to -100 and 100
    bounds = [(-1, 1)]  # Adjusted bounds

    iteration_count = 0
    min_iterations = 50
    max_iterations = 1000
    convergence_iteration = None

    # Initialize the plot
    fig, ax = plt.subplots()
    ax.set_xlim(bounds[0])
    ax.set_ylim([-1, 1])  # Adjust as necessary for your specific problem

    all_solutions = []
    all_objectives = []

    best_solution = None
    best_objective = float('inf')
    best_solution_value = float('inf')  # Track the best solution based on value

    def callback_function(xk, convergence):
        nonlocal iteration_count, convergence_iteration, best_solution, best_objective, best_solution_value
        iteration_count += 1
        # print(f"Iteration: {iteration_count}")
        current_objective = objective_quantile(xk)
        #current_objective = objective_mse(xk)
        print(f"Iteration: {iteration_count}, Current Solution: {xk[0]}, Objective Value: {current_objective}")

        all_solutions.append(xk[0])
        all_objectives.append(current_objective)

        # Update the best solution if current one is better
        if current_objective < best_objective:
            best_solution = xk
            best_objective = current_objective

        # Update the best solution based on the current solution's absolute value
        # In this approach the reconstructed image is better than the other approach (loss comparision), however it is worse after rescaling
        # if abs(xk[0]) < best_solution_value:
        #     best_solution = xk
        #     best_solution_value = abs(xk[0])
        #     best_objective = current_objective


        # Update the plot with the best solution
        ax.clear()
        ax.set_xlim(bounds[0])
        ax.set_ylim([-100, 100])  # Adjust as necessary for your specific problem
        # ax.scatter(xk[0], objective_quantile(xk), color='red', label='Best Solution')
        #ax.scatter(xk[0], current_objective, color='red', label='Best Solution')
        if all_solutions:
            ax.scatter(all_solutions, all_objectives, color='blue', label='All Solutions')
        if best_solution is not None:
            ax.scatter(best_solution[0], best_objective, color='red', label='Best Solution')
        ax.legend()
        plt.draw()
        #plt.pause(0.1)

        # Record the iteration at which convergence was found
        if convergence and convergence_iteration is None:
            convergence_iteration = iteration_count
            print(f"Convergence detected at iteration {iteration_count}")

        # Ensure the loop runs for at least `min_iterations`
        if iteration_count < min_iterations:
            return False  # Continue optimization

        # Stop condition
        if convergence or iteration_count >= max_iterations:
            if convergence:
                print(f"Stopping optimization at iteration {iteration_count} as convergence was already detected at {convergence_iteration}")
            else:
                print(f"Maximum iterations exceeded {iteration_count}")
            return True  # Stop the optimization

        return False


    # Perform Differential Evolution optimization with updated population size
    # result = differential_evolution(objective_quantile, bounds, popsize=10, callback=callback_function, maxiter=max_iterations, polish=False, init='latinhypercube', updating='deferred')
    while iteration_count < min_iterations:
        result = differential_evolution(objective_quantile, bounds, popsize=10, callback=callback_function,
                                        maxiter=max_iterations, polish=False, init='latinhypercube',
                                        updating='deferred')
        # result = differential_evolution(objective_mse, bounds, popsize=10, callback=callback_function,
        #                                 maxiter=max_iterations, polish=False, init='latinhypercube',
        #                                 updating='deferred')
        if result.success and iteration_count >= min_iterations:
            best_solution = result.x
            best_objective = result.fun
            break

    #u_optimized = result.x[0]
    u_optimized = best_solution[0]

    # Calculate the final predicted udldu
    u = torch.tensor(u_optimized).to(**setup)
    udldu_ = -u / (1 + torch.exp(u))

    print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
    print(f"Optimal solution: {u_optimized}")
    print(f"Bounds used: {bounds}")
    print(f"Total number of iterations performed: {iteration_count}")
    print(f"Total number of iterations required: {result.nit}")

    # Final plot display
    # plt.show()
    save_path= "Plots/Img-5.png"
    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")

    return u_optimized

       # def callback_function(xk, convergence):
    #     nonlocal iteration_count
    #     iteration_count += 1
    #     print(f"Iteration: {iteration_count}")
    #
    #     # Update the plot with the best solution
    #     ax.clear()
    #     ax.set_xlim(bounds[0])
    #     ax.set_ylim([-100, 100])  # Adjust as necessary for your specific problem
    #     ax.scatter(xk[0], objective_quantile(xk), color='red', label='Best Solution')
    #     ax.legend()
    #     plt.pause(0.1)
    #
    #     # Print convergence status
    #     if convergence:
    #         print(f"Convergence reached at iteration {iteration_count}")
    #         return True  # Stop the optimization if convergence is detected
    #     elif iteration_count >= min_iterations:
    #         return True  # Stop based on iteration count if convergence has not been reached
    #
    #     return False  # Continue optimization

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

# Original Version (This goes under "setup")
# def logistic_loss(y, pred):
#     y = torch.tensor(y).to(**setup)
#     pred = torch.squeeze(pred, -1)
#     return torch.mean(-(y*torch.log(pred)+(1-y)*torch.log(1-pred)))

# def inverse_udldu(udldu):
#     '''derive u from udldu using gradient descend based method'''
#     lr = 0.01
#     u = torch.tensor(0).to(**setup).requires_grad_(True)
#     udldu = torch.tensor(udldu).to(**setup)
#     optimizer = torch.optim.Adam([u], lr=lr)
#     loss_fn = nn.MSELoss()
#     for i in range(30000):
#         optimizer.zero_grad()
#         udldu_ = -u / (1 + torch.exp(u))
#         l = loss_fn(udldu_, udldu)
#         l.backward()
#         optimizer.step()
#     udldu_ = -u / (1 + torch.exp(u))
#     print(f"The error term of inversing udldu: {udldu.item()-udldu_.item():.1e}")
#     return u.detach().numpy()

# My Version
# def inverse_udldu(udldu):
#     '''derive u from udldu using Differential Evolution method'''
#     udldu = torch.tensor(udldu).to(**setup)
#
#     # Define the objective function for DE
#     # def objective_mse(u):
#     #     u = torch.tensor(u).to(**setup)
#     #     udldu_ = -u / (1 + torch.exp(u))
#     #     loss = torch.mean((udldu_ - udldu) ** 2).item()
#     #     return loss
#     #
#     # def objective_mae(u):
#     #     u = torch.tensor(u).to(**setup)
#     #     udldu_ = -u / (1 + torch.exp(u))
#     #     loss = torch.mean(torch.abs(udldu_ - udldu)).item()
#     #     return loss
#     #
#     # def objective_huber(u, delta=1.0):
#     #     u = torch.tensor(u).to(**setup)
#     #     udldu_ = -u / (1 + torch.exp(u))
#     #     loss = torch.mean(torch.where(torch.abs(udldu_ - udldu) < delta,
#     #                                   0.5 * (udldu_ - udldu) ** 2,
#     #                                   delta * torch.abs(udldu_ - udldu) - 0.5 * delta)).item()
#     #     return loss
#     #
#     # def objective_log_cosh(u):
#     #     u = torch.tensor(u).to(**setup)
#     #     udldu_ = -u / (1 + torch.exp(u))
#     #     loss = torch.mean(torch.log(torch.cosh(udldu_ - udldu))).item()
#     #     return loss
#
#     def objective_quantile(u, quantile=0.5):
#         u = torch.tensor(u).to(**setup)
#         udldu_ = -u / (1 + torch.exp(u))
#         loss = torch.mean(torch.max((quantile - (udldu_ - udldu) < 0) * (quantile - (udldu_ - udldu)),
#                                     (1 - quantile - (udldu_ - udldu) > 0) * (1 - quantile - (udldu_ - udldu)))).item()
#         return loss
#
#     # def objective_chebyshev(u):
#     #     u = torch.tensor(u).to(**setup)
#     #     udldu_ = -u / (1 + torch.exp(u))
#     #     loss = torch.max(torch.abs(udldu_ - udldu)).item()
#     #     return loss
#
#     # Define bounds for u (you can adjust these bounds as needed)
#     bounds = [(-1, 1)]  # Example bounds; adjust as necessary
#     iteration_count = 0
#
#     def callback_function(xk, convergence):
#         nonlocal iteration_count
#         iteration_count += 1
#         print(f"Iteration: {iteration_count}")
#
#     # Perform Differential Evolution optimization
#     result = differential_evolution(objective_quantile, bounds, popsize=7, callback=callback_function, maxiter=100)
#     u_optimized = result.x[0]
#
#     # Calculate the final predicted udldu
#     u = torch.tensor(u_optimized).to(**setup)
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#
#     # Print the total number of iterations after completion
#     print(f"Total number of iterations performed: {result.nit}")
#
#     return u_optimized

# def inverse_udldu(udldu):
#     '''derive u from udldu using LBFGS method'''
#     udldu = torch.tensor(udldu).to(**setup)
#
#     # Define the objective function for LBFGS
#     def objective(u):
#         u = torch.tensor(u).to(**setup)
#         udldu_ = -u / (1 + torch.exp(u))
#         loss = torch.mean((udldu_ - udldu) ** 2).item()
#         return loss
#
#     # Initial guess for u
#     u_init = 0.0  # Adjust initial guess as needed
#
#     # Perform LBFGS optimization
#     result = opt.minimize(objective, x0=u_init, method='L-BFGS-B')
#     u_optimized = result.x[0]
#
#     # Calculate the final predicted udldu
#     u = torch.tensor(u_optimized).to(**setup)
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#     return u_optimized

# def inverse_udldu(udldu):
#     '''derive u from udldu using Adam optimizer'''
#     udldu = torch.tensor(udldu).to(**setup)
#
#     # Initialize u as a tensor with random values
#     u = torch.randn(1, requires_grad=True).to(**setup)
#
#     # Define the optimizer
#     optimizer = torch.optim.Adam([u], lr=0.01)  # Adjust learning rate as needed
#
#     # Set the number of epochs for training
#     num_epochs = 1000  # Adjust as needed
#
#     for epoch in range(num_epochs):
#         # Calculate the predicted udldu
#         udldu_ = -u / (1 + torch.exp(u))
#
#         # Compute the loss
#         loss = torch.mean((udldu_ - udldu) ** 2)
#
#         # Backpropagate the gradients
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Update the parameters using Adam
#         optimizer.step()
#
#     # Calculate the final predicted udldu
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#     return u.item()

# def inverse_udldu(udldu):
#     '''derive u from udldu using RMSProp optimizer'''
#     udldu = torch.tensor(udldu).to(**setup)
#
#     # Initialize u as a tensor with random values
#     u = torch.randn(1, requires_grad=True).to(**setup)
#
#     # Define the optimizer
#     optimizer = torch.optim.RMSprop([u], lr=0.01)  # Adjust learning rate as needed
#
#     # Set the number of epochs for training
#     num_epochs = 1000  # Adjust as needed
#
#     for epoch in range(num_epochs):
#         # Calculate the predicted udldu
#         udldu_ = -u / (1 + torch.exp(u))
#
#         # Compute the loss
#         loss = torch.mean((udldu_ - udldu) ** 2)
#
#         # Backpropagate the gradients
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Update the parameters using RMSProp
#         optimizer.step()
#
#     # Calculate the final predicted udldu
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#     return u.item()

# def inverse_udldu(udldu):
#     '''derive u from udldu using Momentum optimizer'''
#     udldu = torch.tensor(udldu).to(**setup)
#
#     # Initialize u as a tensor with random values
#     u = torch.randn(1, requires_grad=True).to(**setup)
#
#     # Define the optimizer
#     optimizer = torch.optim.SGD([u], lr=0.01, momentum=0.9)  # Adjust learning rate and momentum as needed
#
#     # Set the number of epochs for training
#     num_epochs = 1000  # Adjust as needed
#
#     for epoch in range(num_epochs):
#         # Calculate the predicted udldu
#         udldu_ = -u / (1 + torch.exp(u))
#
#         # Compute the loss
#         loss = torch.mean((udldu_ - udldu) ** 2)
#
#         # Backpropagate the gradients
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Update the parameters using Momentum
#         optimizer.step()
#
#     # Calculate the final predicted udldu
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#     return u.item()

# def inverse_udldu(udldu):
#     '''derive u from udldu using Nesterov Accelerated Gradient (NAG) method'''
#     # Learning rate and initial values
#     lr = 0.01
#     momentum = 0.9  # Momentum parameter for NAG
#
#     # Convert input to tensor and set up for gradient computation
#     u = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(**setup)
#     udldu = torch.tensor(udldu, dtype=torch.float32).to(**setup)
#
#     # Define the optimizer with Nesterov Accelerated Gradient (NAG)
#     optimizer = SGD([u], lr=lr, momentum=momentum, nesterov=True)
#
#     # Define the loss function
#     loss_fn = nn.MSELoss()
#
#     # Optimization loop
#     num_iterations = 30000  # Number of iterations; adjust as necessary
#     for _ in range(num_iterations):
#         optimizer.zero_grad()  # Clear previous gradients
#         udldu_ = -u / (1 + torch.exp(u))  # Compute the forward pass
#         loss = loss_fn(udldu_, udldu)  # Compute the loss
#         loss.backward()  # Compute gradients
#         optimizer.step()  # Update parameters
#
#     # Final prediction and error calculation
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#     return u.detach().numpy()


# def inverse_udldu(udldu):
#     '''derive u from udldu using AdaMax optimization method'''
#     # Learning rate and initial values
#     lr = 0.01
#
#     # Convert input to tensor and set up for gradient computation
#     u = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(**setup)
#     udldu = torch.tensor(udldu, dtype=torch.float32).to(**setup)
#
#     # Define the optimizer with AdaMax
#     optimizer = Adamax([u], lr=lr)
#
#     # Define the loss function
#     loss_fn = nn.MSELoss()
#
#     # Optimization loop
#     num_iterations = 30000  # Number of iterations; adjust as necessary
#     for _ in range(num_iterations):
#         optimizer.zero_grad()  # Clear previous gradients
#         udldu_ = -u / (1 + torch.exp(u))  # Compute the forward pass
#         loss = loss_fn(udldu_, udldu)  # Compute the loss
#         loss.backward()  # Compute gradients
#         optimizer.step()  # Update parameters
#
#     # Final prediction and error calculation
#     udldu_ = -u / (1 + torch.exp(u))
#
#     print(f"The error term of inversing udldu: {udldu.item() - udldu_.item():.1e}")
#     return u.detach().numpy()

