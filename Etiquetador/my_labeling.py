__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import utils_data as ud
import numpy as np
from PIL import Image
from Kmeans import *
from KNN import *
import time as t
import matplotlib.pyplot as plt

def Retrieval_by_color(list_img, color_labels, color_question):
    
    trobats = []
    
    for i in range(len(list_img)):
        if isinstance(color_question, str):
            for j in range(len(color_labels[i])):
                if color_labels[i][j] == color_question:
                    trobats.append(list_img[i])
        else:
            for color in color_question:
                for j in range(len(color_labels[i])):
                    if color_labels[i][j] == color_question:
                        trobats.append(list_img[i])
    
    return trobats

def Retrieval_by_shape(list_img, shape_labels, shape_question):
    
    trobats = []
    
    for i in range(len(list_img)):
        if shape_labels[i] == shape_question:
            trobats.append(list_img[i])
    
    return trobats


def Retrieval_combined(list_img, shape_labels, color_labels, shape_question, color_question):
    
    trobats = []
    
    for i in range(len(list_img)):
        if shape_labels[i] == shape_question:
            for j in range(len(color_labels[i])):
                if color_labels[i][j] == color_question:
                    trobats.append(list_img[i])
    
    return trobats

def Kmean_statistics(Kmeans_list, Kmax):
    WCD = np.zeros((len(Kmeans_list)))
    iters = np.zeros((len(Kmeans_list)))
    time = np.zeros((len(Kmeans_list)))
    
    for i in range(len(Kmeans_list)):
        WCD_list = []
        time_list = []
        Kmeans = Kmeans_list[i]
        for j in range(2, Kmax):
            Kmeans.K = j
            inici = t.process_time()
            Kmeans.fit()
            final = t.process_time()
            temps = inici - final 
            time_list.append(temps)
            Kmeans.within_class_distance()
            WCD_list.append(Kmeans.WCD)
        WCD[i] = WCD_list
        time[i] = time_list
        iters[i] = Kmeans.num_iter
        
    return WCD, time, iters


def mostrar_imagenes(imagenes):
    num_imagenes = len(imagenes)
    fig = plt.figure(figsize=(12, 8))


    for i, imagen in enumerate(imagenes):
        ax = fig.add_subplot(1, num_imagenes, i+1)
        ax.imshow(imagen, cmap='gray')  # Utiliza cmap='gray' si las imágenes son en escala de grises
        ax.axis('off')

    plt.tight_layout()
    plt.show()

            
            
    
if __name__ == '__main__':
    
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = ud.read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    

    test_imgs = test_imgs[:100]
    knn = KNN(train_imgs, train_class_labels)
    color_results = []
    label_results = knn.predict(test_imgs, 10)
    Kmeans = []
    for image in test_imgs:
        km = KMeans(image, 7)
        km.fit()    
        Kmeans.append(km)
        colors = get_colors(np.array([list(km.centroids[0]), list(km.centroids[1]), list(km.centroids[2])]))
        color_results.append(colors)
    
    for Kmean in Kmeans:
        ax = ud.Plot3DCloud(Kmean)
    
    
    print("Mostrando pantalones negros")
    pantalones_negros = Retrieval_combined(test_imgs, label_results, color_results, "Jeans", "Black")
    ud.visualize_retrieval(pantalones_negros, 1)
    #mostrar_imagenes(pantalones_negros)
    
    print("Mostrando ropa verde")
    ropa_verde = Retrieval_by_color(test_imgs[:100], color_results,"Green")
    ud.visualize_retrieval(ropa_verde, 1)
    #mostrar_imagenes(ropa_verde)
    
    print("Mostrando vestidos")
    vestidos = Retrieval_by_shape(test_imgs[:100], label_results,"Dresses")
    ud.visualize_retrieval(vestidos, 1)
    #mostrar_imagenes(vestidos)
    
    for Kmean in Kmeans:
        ud.visualize_k_means(Kmean, (4800,1))
    """
    WCD, time, iters = Kmean_statistics(Kmeans, 7)
    for wcd, times in zip(WCD, time):
        print(wcd, times)
    print(iters)
    """
    
    # You can start coding your functions here