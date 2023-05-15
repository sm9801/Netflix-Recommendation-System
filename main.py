import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_incomplete.txt")

# TODO: Your code here

X_gold = np.loadtxt('netflix_complete.txt')

mixture, post = common.init(X, 12, 1)
mixture, post, LL = em.run(X, mixture, post)
print(LL)
X_pred = em.fill_matrix(X, mixture)
print(common.rmse(X_gold, X_pred))

# mixture, post = common.init(X, 3, 0)
# mixture, post, cost = naive_em.run(X, mixture, post)
#
#
# K = 1, seed = 0
# K = 2, seed = 0
# K = 3, seed = 3
# K = 4, seed = 4
# mixture1, post1 = common.init(X, 4, 4)
# mixture1, post1, cost1 = kmeans.run(X, mixture1, post1)

# plot_em1 = common.plot(X, mixture, post, 'K = 2, seed = 0, Mixture' )
# plot_k1 = common.plot(X, mixture1, post1, 'K = 2, seed = 0, Kmeans')

# m = common.bic(X, mixture, cost)
# print(m)