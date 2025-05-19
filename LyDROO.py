# -*- coding: utf-8 -*-
"""
Modified LyDROO.py
Runs the LyDROO simulation for a range of IoT device user counts (from 5 to 30).
References:
[1] Bi et al., "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading..."
[2] Huang et al., "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks..."
[3] Bi and Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation offloading”
"""

import scipy.io as sio      # for saving .mat files
import numpy as np
import math

# Import the deep RL memory network (PyTorch-based)
from memory import MemoryDNN
# Import the resource allocation algorithm
from ResourceAllocation import Algo1_NUM

def plot_rate(rate_his, rolling_intv=50, ylabel='Normalized Computation Rate'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(np.arange(len(rate_array)) + 1,
             np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1,
                     np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
                     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values),
                     color='b', alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()

def racian_mec(h, factor):
    n = len(h)
    beta = np.sqrt(h * factor)         # LOS component
    sigma = np.sqrt(h * (1 - factor) / 2)  # scattering standard deviation
    x = sigma * np.random.randn(n) + beta
    y = sigma * np.random.randn(n)
    g = np.power(x, 2) + np.power(y, 2)
    return g

# --- Simulation Settings ---
LOWER_BOUND_USERS = 5
UPPER_BOUND_USERS = 30
STEP_USERS = 1

n = 5000  # number of time frames per simulation

for N in range(LOWER_BOUND_USERS, UPPER_BOUND_USERS + 1, STEP_USERS):
    print("===================================================")
    print("Starting simulation for N = {} users".format(N))
    
    # Candidate action count initially equals number of users
    K = N  
    
    # Simulation configuration parameters:
    decoder_mode = 'OPN'    # quantization mode: 'OP', 'KNN', or 'OPN'
    Memory_capacity = 4096  # Increased memory capacity for more users
    Delta = 32              # Update interval for adaptive candidate count K
    CHFACT = 10**10         # Scaling factor for channel values
    
    # Energy consumption threshold per user (scaled to reflect increased capacity)
    energy_thresh = np.ones((N)) * 0.08 * 4
    nu = 1000             # Energy queue factor
    # User weighting (alternating pattern)
    w = [1.5 if i % 2 == 0 else 1 for i in range(N)]
    V = 80  # Increased control parameter
    
    # Data arrival rate per user (in Mbps)
    arrival_lambda = 3 * np.ones((N))
    
    print('#users = %d, #time frames = %d, K=%d, decoder=%s, Memory=%d, Delta=%d' %
          (N, n, K, decoder_mode, Memory_capacity, Delta))
    
    # Initialize simulation arrays (dimensions: [n, N])
    channel = np.zeros((n, N))
    dataA = np.zeros((n, N))
    Q = np.zeros((n, N))          # Data queue (in Mbits)
    Y = np.zeros((n, N))          # Virtual energy queue (in mJ)
    Obj = np.zeros(n)             # Objective values per time frame
    energy = np.zeros((n, N))     # Energy consumption
    rate = np.zeros((n, N))       # Computation rate
    
    # Set up channel parameters (each user has a unique distance)
    dist_v = np.linspace(start=120, stop=255, num=N)
    Ad = 3
    fc = 915 * 10**6
    loss_exponent = 3
    light = 3 * 10**8
    h0 = np.ones((N))
    for j in range(N):
        h0[j] = Ad * (light / (4 * math.pi * fc * dist_v[j])) ** loss_exponent
    
    # Initialize the MemoryDNN with input dimension = N*3 (features: channel, normalized Q, normalized Y)
    mem = MemoryDNN(net=[N * 3, 256, 128, N],
                    learning_rate=0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory_capacity)
    
    mode_his = []   # Record chosen offloading actions over time
    k_idx_his = []  # Record index of best candidate action per time frame
    
    # Main simulation loop
    for i in range(n):
        if i % (n // 10) == 0:
            print("N =", N, "Progress: %0.1f%%" % (100 * i / n))
        
        # Update candidate action count adaptively every Delta frames
        if i > 0 and i % Delta == 0:
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1]) % K) + 1
            else:
                max_k = k_idx_his[-1] + 1
            K = min(max_k + 1, N)
        
        # Generate channel gains for current frame using Rician fading and scale
        h_tmp = racian_mec(h0, 0.3)
        h = h_tmp * CHFACT
        channel[i, :] = h
        
        # Generate data arrivals for current frame
        dataA[i, :] = np.random.exponential(arrival_lambda)
        
        # Queueing Module: Update data and energy queues
        if i > 0:
            Q[i, :] = Q[i - 1, :] + dataA[i - 1, :] - rate[i - 1, :]
            Q[i, Q[i, :] < 0] = 0
            Y[i, :] = np.maximum(Y[i - 1, :] + (energy[i - 1, :] - energy_thresh) * nu, 0)
            Y[i, Y[i, :] < 0] = 0
        
        # Form input for the DNN by concatenating channel, normalized Q, and normalized Y
        nn_input = np.concatenate((h, Q[i, :] / 10000, Y[i, :] / 10000))
        
        # Actor Module: Generate candidate offloading actions using MemoryDNN
        m_pred, m_list = mem.decode(nn_input, K, decoder_mode)
        r_list = []  # Store resource allocation outcomes for each candidate action
        v_list = []  # Store corresponding objective values
        
        # Critic Module: Evaluate each candidate action using resource allocation (Algo1_NUM)
        for m in m_list:
            res = Algo1_NUM(m, h, w, Q[i, :], Y[i, :], V)
            r_list.append(res)
            v_list.append(res[0])
        
        # Select the candidate action with the highest objective value
        best_idx = np.argmax(v_list)
        k_idx_his.append(best_idx)
        mem.encode(nn_input, m_list[best_idx])
        mode_his.append(m_list[best_idx])
        
        # Record objective value, computation rate, and energy consumption for current frame
        Obj[i], rate[i, :], energy[i, :] = r_list[best_idx]
    
    # Plot training cost history from MemoryDNN
    mem.plot_cost()
    
    # Plot average data queue and average energy consumption vs. time frames
    plot_rate(Q.sum(axis=1) / N, 100, 'Average Data Queue')
    plot_rate(energy.sum(axis=1) / N, 100, 'Average Energy Consumption')
    
    # Save simulation results to a .mat file (filename indicates the user count)
    filename = './result_%d.mat' % N
    sio.savemat(filename, {
        'input_h': channel / CHFACT,
        'data_arrival': dataA,
        'data_queue': Q,
        'energy_queue': Y,
        'off_mode': mode_his,
        'rate': rate,
        'energy_consumption': energy,
        'data_rate': rate,
        'objective': Obj
    })
    
    print("Results for N = {} users saved in {}".format(N, filename))
