import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm


def probit_vector(u_vec):
    return np.array([stats.norm.cdf(u,0,1) for u in u_vec])

def make_data_and_gibbs(w, mu, sigma, N=100, B_burn = 1000, B_save = 500):
    ######### Synthesize data
    x = np.random.randn(N,D)
    y = np.squeeze(np.array([np.random.binomial(1, p) for p in probit_vector(np.matmul(w.reshape((1,2)),x.T))]))
    ######### Done synthesizing data

    ######### Gibbs Sampling
    w_gibbs_list = []

    # 0. Initialize w
    w_gibbs = np.random.randn(D)

    for b in tqdm(range(B_burn+B_save)):
        # 1. Sample z_sampled[n] ~ p(z_n|x[n],y[n],w_gibbs) for n=1,...,N
        z_sampled = np.empty(N)
        for n in range(N):
            condition_satisfied = False
            while not condition_satisfied:
                z_sampled[n] = np.random.normal(np.dot(x[n],w_gibbs), 1)
                if y[n] == 1:
                    condition_satisfied = z_sampled[n] >= 0
                else:
                    condition_satisfied = z_sampled[n] < 0
        # z_sampled

        # 2. Sample w_gibbs ~ p(w|{x[n],y[n],z_sampled[n]})
        J = np.linalg.inv(sigma) + np.matmul(x.T,x)
        sigma_cond_gibbs = np.linalg.inv(J)
        h = np.matmul(np.linalg.inv(sigma), mu)
        for n in range(N):
            h += z_sampled[n]*x[n]
        w_gibbs = np.random.multivariate_normal(np.matmul(sigma_cond_gibbs,h), sigma_cond_gibbs)

        # 3. Store w_gibbs if bn >= B_burn
        if b >= B_burn:
            w_gibbs_list.append(w_gibbs)

    return w_gibbs_list

def try_Ns_and_plot(w, mu, sigma, Ns=[100, 500], B_burn = 1000, B_save = 500):
    f,ax = plt.subplots(1,len(Ns), figsize=(9,3))

    for i in range(len(Ns)):
        N = Ns[i]
        w_gibbs_list = make_data_and_gibbs(w, mu, sigma, N, B_burn, B_save)
        w_gibbs_xs = [w_g[0] for w_g in w_gibbs_list]
        w_gibbs_ys = [w_g[1] for w_g in w_gibbs_list]
        ax[i].scatter(w_gibbs_xs, w_gibbs_ys, c='blue')
        ax[i].scatter(*w, c='red')
        ax[i].set_aspect('equal')
        ax[i].set_xlim([w[0]-.5, w[0]+.5])
        ax[i].set_ylim([w[1]-.5, w[1]+.5])
        ax[i].set_title("N = " + str(N))
    f.show()
    return f



######### Synthesize params
D = 2
mu = np.zeros(D)
temp = np.random.randn(D,D)
sigma = temp.T * temp
w = np.random.multivariate_normal(mu, sigma)

######## Run exps and plot
f = try_Ns_and_plot(w, mu, sigma, [100,300,500])
