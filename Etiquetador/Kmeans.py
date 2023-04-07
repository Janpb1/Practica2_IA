__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        x = np.array(X)
        
        if x.dtype is not float:
            x = x.astype(float)
        
        if x.dim > 2:
            tamany = x.shape
            x = x.reshape(tamany[0]*tamany[1],3)
            # x = x.reshape((tamany[0]*tamany[1],tamany[2]))
            
            #elements = x.prod(x.shape)
            #x = x.reshape(elements//3, 3)
        
        self.X = x
        
        

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################   
            


def _init_centroids(self):
    """
    Initialization of centroids
    """

    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    
    if self.options['km_init'].lower() == 'first':
        self.first_centroid()
    elif self.options['km_init'].lower() == 'random':
        self.random_centroid()
    elif self.options['km_init'].lower() == 'custom':
        self.custon_centroid()
    else:
        self.centroids = np.random.rand(self.K, self.X.shape[1])
        self.old_centroids = np.random.rand(self.K, self.X.shape[1])

def first_centroid(self):
    self.centroids.append(self.X[0]) #Inicialitzem el centroide amb el primer pixel
    centroide_iniciats = 1 
    while centroide_iniciats < self.K: #Comparem amb self.K ja que es el numero de centroides
        for pixel in self.X:
            repetit = False
            for centroids in self.centroids:
                if np.array_equal(pixel, centroids): #Comparem que no sigui un centroide ja agafat
                    repetit = True 
            if not repetit:
                self.centroids.append(pixel)
                centroide_iniciats += 1
                break


def random_centroid(self):

    self.centroids.append(np.random.choice(self.X.flatten())) #Inicialitzem el centroide amb un pixel aleatori
    centroide_iniciats = 1 
    while centroide_iniciats < self.K: #Comparem amb self.K ja que es el numero de centroides
        centroide_aleatori = self.centroids.append(np.random.choice(self.X.flatten())) #Fem servir el flatten per posar tots els valors en un array
        if centroide_aleatori not in self.centroide: #Comparem que no sigui un centroide ja agafat
            self.centroids.append(centroide_aleatori)
            centroide_iniciats += 1


def custom_centroid(self):

    #Utilitzarem el KMeans++ per a la busqueda dels centroids com a opció random

    self.centroids.append(np.random.choice(self.X.flatten())) #Inicialitzem amb un centroid aleatori
    centroide_iniciats = 1 
    while centroide_iniciats < self.K:
        distancies = np.array([min([np.linalg.norm(x-c) for c in self.centroids]) for x in self.X]) #Es calcula les distancies de tots els punts fins al centroid escollit
        probabilitats = distancies / np.sum(distancies)
        self.centroids.append(np.random.choice((self.X.flatten()),p=probabilitats)) #S'escull el centroid segons la probabilitat de la distancia entre el nou i el vell centroid


def get_labels(self):
    """        Calculates the closest centroid of all points in X
    and assigns each point to the closest centroid
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    self.labels = np.random.randint(self.K, size=self.X.shape[0])


def get_centroids(self):
    """
    Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    pass


def converges(self):
    """
    Checks if there is a difference between current and old centroids
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    return True


def fit(self):
    """
    Runs K-Means algorithm until it converges or until the number
    of iterations is smaller than the maximum number of iterations.
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    pass


def withinClassDistance(self):
    """
     returns the within class distance of the current clustering
    """

    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    return np.random.rand()


def find_bestK(self, max_K):
    """
     sets the best k anlysing the results up to 'max_K' clusters
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    pass


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    dist = np.empty()
    for point in X:
        point = np.array(point)
        dist_points = np.empty()
        for centroid in C:
            centroid = np.array(centroid)
            dist_points.append(np.linalg.norm(point - centroid))
        dist.append(dist_points)
    return dist




def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
