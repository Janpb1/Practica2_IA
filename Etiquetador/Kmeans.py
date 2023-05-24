__authors__ = ['1639484','1636492','1638248']
__group__ = 'DJ.12'

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
        self.centroids = np.zeros((K, self.X.shape[1]))
        self.old_centroids = np.zeros((K, self.X.shape[1]))
        self.labels = np.zeros((K, self.X.shape[1]))
        #self.WCD = 0
        

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        
        x = np.array(X)
        
        if x.dtype is not float:
            x = x.astype(float)

        if x.ndim > 2:
            tamany = x.shape
            x = x.reshape(tamany[0]*tamany[1],3)
            
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
        

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            self.first_centroid()
        elif self.options['km_init'].lower() == 'random':
            self.random_centroid()
        elif self.options['km_init'].lower() == 'custom':
            self.custom_centroid()
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])


    def first_centroid(self):
        """
        #self.centroids = np.zeros((self.K, self.X.shape[1]))
        selected_pixels = {}
        centroides = []
        centroides_iniciats = 0
        while centroides_iniciats < self.K:
            for pixel in self.X:
                aux = tuple(pixel)
                if aux not in selected_pixels:
                    centroides.append(pixel)
                    selected_pixels[aux] = 1  # Marcando el píxel como usado para no repetirlo
                    break
        
        self.centroids = np.array(self.centroids)
        self.old_centroids = np.copy(self.centroids)

        """
        selected_pixels = {}
        self.centroids = []
        for i in range(self.K):
            for pixel in self.X:
                aux = tuple(pixel)
                if aux not in selected_pixels:
                    self.centroids.append(pixel)
                    selected_pixels[aux] = 1  # Marking the pixel as used so we don't repeat it
                    break
        self.centroids = np.copy(self.centroids)
        self.old_centroids = np.copy(self.centroids)
        

    def random_centroid(self):
        self.centroids[0] = np.random.choice(self.X.flatten()) #Inicialitzem el centroide amb un pixel aleatori
        centroide_iniciats = 1 
        while centroide_iniciats < self.K: #Comparem amb self.K ja que es el numero de centroides
            centroide_aleatori = np.random.choice(self.X.flatten()) #Fem servir el flatten per posar tots els valors en un array
            if centroide_aleatori not in self.centroide: #Comparem que no sigui un centroide ja agafat
                self.centroids[centroide_iniciats] = centroide_aleatori
                centroide_iniciats += 1
        self.old_centroids = self.centroids

    def custom_centroid(self):
        #Utilitzarem el KMeans++ per a la busqueda dels centroids com a opció random
        self.centroids.append(np.random.choice(self.X.flatten())) #Inicialitzem amb un centroid aleatori
        centroide_iniciats = 1 
        while centroide_iniciats < self.K:
            distancies = np.array([min([np.linalg.norm(x-c) for c in self.centroids]) for x in self.X]) #Es calcula les distancies de tots els punts fins al centroid escollit
            probabilitats = distancies / np.sum(distancies)
            self.centroids.append(np.random.choice((self.X.flatten()),p=probabilitats)) #S'escull el centroid segons la probabilitat de la distancia entre el nou i el vell centroid
        
        self.old_centroids = self.centroids


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        distancias = np.zeros((self.X.shape[0], self.K))
        for cluster in range(self.K):
            distancias[:, cluster] = np.linalg.norm(self.X - self.centroids[cluster], axis=1)
        self.labels = np.argmin(distancias, axis=1)
        

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids
        nous_centroides = [[] for x in range(self.K)]
        for point in range(len(self.X)):
            centroide = self.labels[point] #Agafim el centroide que li pertoca al punt calculat a l'atribut labels
            nous_centroides[centroide].append(self.X[point]) #Afegim el punt al centroid que li pertoca
            
        for centroides in range(len(nous_centroides)):
            nous_centroides[centroides] = np.average(np.array(nous_centroides[centroides]), 0)
        
        self.centroids = np.array(nous_centroides)
            

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        if self.centroids.size == self.old_centroids.size:
            iguals = np.allclose(self.centroids, self.old_centroids, rtol = self.options['tolerance'], atol = self.options['tolerance'], equal_nan = False)
        else:
            iguals = False
        return iguals
       

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        i = 0
        self.get_labels()
        self.get_centroids()
        while i < len(self.X) and not self.converges():  
            self.get_labels()
            self.get_centroids()
            i+=1


    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        #self.WCD  
        summation = 0 #Sumatorio
        for point in range(len(self.X)):
            #Agafem el centroide que li pertoca al punt calculat a l'atribut labels i el punt que li correspon dins de l'atribut X
            summation += np.linalg.norm(np.array(self.X[point]) - np.array(self.centroids[self.labels[point]]))**2
        self.WCD = (1/len(self.X)) * summation  
        
        
    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        #self.WCD  
        summation = 0 #Sumatorio
        for point in range(len(self.X)):
            #Agafem el centroide que li pertoca al punt calculat a l'atribut labels i el punt que li correspon dins de l'atribut X
            summation += np.linalg.norm(np.array(self.X[point]) - np.array(self.centroids[self.labels[point]]))**2
        return (1/len(self.X)) * summation #WCD
        
    def interClassDistance(self):
        summation = 0 
        for point in range(len(self.X)):
            #Agafem el centroide que li pertoca al punt calculat a l'atribut labels i el punt que li correspon dins de l'atribut X
            summation += np.linalg.norm(np.array(self.X[point]) - np.array(self.centroids[self.labels[point]]))**2
        return summation / len(self.X) #ICD
            
    def fisher(self):
        return self.withinClassDistance() / self.interClassDistance() #Fisher
    
    def find_bestK(self, max_K, heuristic = 'WCD'):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        self.K = 2
        self.fit()
        if heuristic == 'WCD':
            wcd = self.withinClassDistance()
        elif heuristic == 'ICD':
            icd = self.interClassDistance()
        elif heuristic == 'FISHER':
            fisher = self.fisher()
            
        while self.K < max_K:
            self.K += 1
            self.fit()
            if heuristic == 'WCD':
                DEC = 100 * (wcd / self.withinClassDistance())
            elif heuristic == 'ICD':
                DEC = 100 * (icd / self.interClassDistance())
            elif heuristic == 'FISHER':
                DEC = 100 * (fisher / self.fisher())
                
            #DEC = 100*(self.WCD/wcd)
            #print('WDC: ', self.WCD, 'wdc: ', wcd, '\n')
            #print('DEC: ', DEC, '100-DEC = ', 100-DEC, '\n')
            
            if (100 - DEC) <= 20: #20% llindar per determinar estabilització
                self.K -= 1
                break
        self.fit()
            
        

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
    dist = []
    for point in X:
        point = np.array(point)
        dist_points = []
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
    probabilitats = utils.get_color_prob(centroids) #prob dels colors de cada centroide
    colors = []
    
    #porbabilitats -> matriu amb les porbabilitats de cada centroide
    #Amb argmax agafem l'index maxim de cada fila y amb la funcio colors de la llibreria utils obtenim el color.
    prob = np.argmax(probabilitats, axis=1)
    colors = utils.colors[prob]
    return colors
