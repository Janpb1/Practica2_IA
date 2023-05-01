__authors__ = ['1639484', '1636492', '1638248']
__group__ = 'DJ.12'

from ctypes import POINTER, resize
#from typing import Self
import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)   
        self.neighbors = []


    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if not isinstance(train_data, float):
            train_data = train_data.astype(float)
        
        n_dimensions = train_data.shape
        P = n_dimensions[0]
        D = n_dimensions[1] * n_dimensions[2]
        self.train_data = np.reshape(train_data, (P, D))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        
        n_dimensions = test_data.shape
        N = n_dimensions[0]
        D = n_dimensions[1] * n_dimensions[2]
        NxD = np.reshape(test_data, (N, D))
        dist = cdist(NxD, self.train_data)
        
        for point in range(N):
            dist_point_index = np.argsort(np.array(dist[point]))[:k]
            self.neighbors.append([self.labels[i] for i in dist_point_index])
        self.neighbors = np.array(self.neighbors)
              
                
    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        vots = []
        percentatges = []
        filas = self.neighbors.shape
        for i in range(0, filas[0]):
            clases = [0 for k in list(set(self.labels))]
            sum = 0
            indices = []
            for j in range(0, filas[1]):
                sum += 1
                labels = np.array(list(set(self.labels)))
                index = np.where(labels == self.neighbors[i][j])
                clases[index[0][0]] += 1
                indices.append(index[0][0])
            count = 0
            for clase in clases:
                if max(clases) == clase:
                    count += 1
            if (count > 1):
                conteo = {}
                for elemento in indices:
                    if elemento in conteo:
                        conteo[elemento] += 1
                    else:
                        conteo[elemento] = 1

                elemento_mas_comun = None
                frecuencia_maxima = 0
                for elemento, frecuencia in conteo.items():
                    if frecuencia > frecuencia_maxima:
                        elemento_mas_comun = elemento
                        frecuencia_maxima = frecuencia
                        
                        
                vots.append(labels[elemento_mas_comun])
            else:
                vots.append(list(set(self.labels))[clases.index(max(clases))])
            percentatges.append(round((max(clases)/sum)*100, 2))

        return np.array(vots)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()