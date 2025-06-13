import numpy as np
from scipy.special import lambertw
from scipy.optimize import linprog

def Algo1_NUM(mode, h, w, Q, Y, num_users,V=20):
    ch_fact = 10**10   # scaling factor
    d_fact = 10**6
    Y_factor = 10
    Y = Y * Y_factor
    phi = 100          # CPU cycles per bit
    W = 2              # Bandwidth (MHz)
    k_factor = (10**(-26)) * (d_fact**3)
    vu = 1.1
    
    N0 = W * d_fact * (10**(-17.4)) * (10**(-3)) * ch_fact  # Noise power (W)
    P_max = 0.1        # Maximum transmit power (W)
    f_max = 1200       # Maximum local computing frequency (MHz)
    
    N = num_users
    if len(w) == 0:
        w = np.ones((N))
    
    # Control parameter a for each user
    a = np.ones((N))
    q = Q
    for i in range(N):
        a[i] = Q[i] + V * w[i]
       
    energy = np.zeros((N))
    rate = np.zeros((N))
    f0_val = 0
    
    # Local computing (mode==0)
    idx0 = np.where(mode == 1)[0]
    M0 = len(idx0)

    # print(f"idx0 : {idx0}")
    # print(f"Y : {Y}")

    if M0 != 0:
        Y0 = np.zeros((M0))
        a0 = np.zeros((M0))
        q0 = np.zeros((M0))
        f0 = np.zeros((M0))
        for i in range(M0):
            tmp_id = idx0[i]
            Y0[i] = Y[tmp_id]
            a0[i] = a[tmp_id]
            q0[i] = q[tmp_id]
            if Y0[i] == 0:
                f0[i] = np.minimum(phi * q0[i], f_max)
            else:
                tmp1 = np.sqrt(a0[i] / (3 * phi * k_factor * Y0[i]))
                tmp2 = np.minimum(phi * q0[i], f_max)
                f0[i] = np.minimum(tmp1, tmp2)
            energy[tmp_id] = k_factor * (f0[i] ** 3)
            rate[tmp_id] = f0[i] / phi
            f0_val += a0[i] * rate[tmp_id] - Y0[i] * energy[tmp_id]
    
    # Offloading (mode==1)
    idx1 = np.where(mode == 1)[0]
    M1 = len(idx1)
    if M1 == 0:
        f1_val = 0
    else:
        Y1 = np.zeros((M1))
        a1 = np.zeros((M1))
        q1 = np.zeros((M1))
        h1 = np.zeros((M1))
        R_max = np.zeros((M1))
        tau1 = np.zeros((M1))
        delta0 = 1
        lb = 0
        ub = 10**4
        
        for i in range(M1):
            tmp_id = idx1[i]
            Y1[i] = Y[tmp_id]
            a1[i] = a[tmp_id]
            q1[i] = q[tmp_id]
            h1[i] = h[tmp_id]
            SNR = h1 / N0
            R_max[i] = W / vu * np.log2(1 + SNR[i] * P_max)
        
        rat = np.zeros((M1))
        e_ratio = np.zeros((M1))
        parac = np.zeros((M1))
        c = np.zeros((M1))
                   
        while np.abs(ub - lb) > delta0:
            mu = (lb + ub) / 2
            for i in range(M1):
                if Y1[i] == 0:
                    rat[i] = R_max[i]
                else:
                    A = 1 + mu / (Y1[i] * P_max)
                    A = np.minimum(A, 20)
                    tmpA = np.real(lambertw(-A * np.exp(-A)))
                    tmp1 = np.minimum(-A / tmpA, 10**20)
                    snr0 = 1 / P_max * (tmp1 - 1)
                    if SNR[i] <= snr0:
                        rat[i] = R_max[i]
                    else:
                        z1 = np.exp(-1) * (mu * SNR[i] / Y1[i] - 1)
                        rat[i] = (np.real(lambertw(z1)) + 1) * W / (np.log(2) * vu)
                e_ratio[i] = 1 / SNR[i] * (2**(rat[i] * vu / W) - 1)
                parac[i] = a1[i] - mu / rat[i] - (Y1[i] * e_ratio[i]) / rat[i]
                c[i] = q1[i] if parac[i] > 0 else 0
                tau1[i] = c[i] / rat[i]
          
            if np.sum(tau1) > 1:
                lb = mu
            else:
                ub = mu

        para_e = np.zeros((M1))
        para   = np.zeros((M1))
        d = np.zeros((M1))
        tau_fact = np.zeros((M1))
        A_matrix = np.zeros((2 * M1 + 1, M1))
        b = np.zeros((2 * M1 + 1))
    
        for i in range(M1):
            para_e[i] = Y1[i] * e_ratio[i] / rat[i]
            para[i] = a1[i] - para_e[i]
            d[i] = q1[i]
            tau_fact[i] = 1 / rat[i]
     
        A_matrix[0:M1, :] = np.eye(M1, dtype=int)
        A_matrix[M1:2 * M1, :] = -np.eye(M1, dtype=int)
        A_matrix[2 * M1, :] = tau_fact
        b[0:M1] = d
        b[M1:2 * M1] = np.zeros((M1))
        b[2 * M1] = 1

        res = linprog(-para, A_ub=A_matrix, b_ub=b)
        r1 = np.maximum(res.x, 0)
        r1 = np.around(r1, decimals=6)
    
        tau1 = np.zeros((M1))
        f1_val = 0
        for i in range(M1):
            tmp_id = idx1[i]
            tau1[i] = r1[i] / rat[i]
            rate[tmp_id] = r1[i]
            energy[tmp_id] = e_ratio[i] * tau1[i]
            f1_val += a1[i] * rate[tmp_id] - Y1[i] * energy[tmp_id]
        
    f_val = f0_val + (f1_val if M1 != 0 else 0)
    f_val = np.around(f_val, decimals=6)
    rate = np.around(rate, decimals=6)
    energy = np.around(energy, decimals=6)
    
    return f_val, rate, energy
