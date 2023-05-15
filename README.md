# Netflix-Recommendation-System

This project uses a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. The goal of this project is to recommend users with movies and shows they might enjoy via collaborative filtering. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. Each user is highly likely to have only rated a small fraction of the movies, and hence, the data matrix is only partially completed. Therefore, the objective is to predict all the remaining entries of the matrix. 

We will implement an Expectation-Maximization (EM) algorithm and use mixtures of Gaussians to achieve our goals. The EM algorithm proceeds by iteratively assigning users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.
