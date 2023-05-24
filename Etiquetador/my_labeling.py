__authors__ = ['1639484','1636492','1638248']
__group__ = 'DJ.12'

import utils_data as ud
import numpy as np
from PIL import Image
from Kmeans import *
from KNN import *
import time as t
import matplotlib.pyplot as plt

""" INICI ANALISI QUALITATIU """

def Retrieval_by_color(list_img, color_labels, color_question):
    trobats = []
    for i in range(len(list_img)):
        for j in range(len(color_labels[i])):
            if color_labels[i][j] in color_question:
                trobats.append(list_img[i])
    return trobats

def Retrieval_by_shape(list_img, shape_labels, shape_question):
    trobats = []    
    for i in range(len(list_img)):
        if shape_labels[i] in shape_question:
            trobats.append(list_img[i])
    return trobats


def Retrieval_combined(list_img, shape_labels, color_labels, shape_question, color_question):
    trobats = []
    for i in range(len(list_img)):
        if shape_labels[i] in shape_question:
            for j in range(len(color_labels[i])):
                if color_labels[i][j] in color_question:
                    trobats.append(list_img[i])
    return trobats

""" FI ANALISI QUALITATIU """

""" INICI ANALISI QUANTITATIU """

def Kmean_statistics(Kmeans_list, Kmax):
    WCD = []
    iters = []
    time = []
    
    for i in range(len(Kmeans_list)):
        WCD_list = []
        time_list = []
        Kmeans = Kmeans_list[i]
        for j in range(2, Kmax):
            Kmeans.K = j
            inici = t.process_time()
            Kmeans.fit()
            final = t.process_time()
            temps = final - inici 
            time_list.append(temps)
            Kmeans.withinClassDistance()
            WCD_list.append(Kmeans.WCD)
        WCD.append(WCD_list)
        time.append(time_list)
        iters.append(Kmeans.num_iter)
        
    return WCD, time, iters

def get_shape_accuracy(classes, gt):
    return str(sum(1 for x, y in zip(labels, gt) if x == y)/len(labels))


def get_color_accuracy(colors, gt):
    color_accuracy = 0
    for i in range(len(labels)):
        labels_set = list(set(labels[i]))
        for j in range(len(set(labels_set))):
            if labels_set[j] in gt[i]:
                color_accuracy += 1/len(gt[i])
    return str(color_accuracy)

""" FI ANALISI QUANTITATIU """

def mostrar_imagenes(imagenes):
    num_imagenes = len(imagenes)
    fig = plt.figure(figsize=(12, 8))


    for i, imagen in enumerate(imagenes):
        ax = fig.add_subplot(1, num_imagenes, i+1)
        ax.imshow(imagen, cmap='gray')  # Utiliza cmap='gray' si las imágenes son en escala de grises
        ax.axis('off')

    plt.tight_layout()
    plt.show()

            
def test_qualitatiu(class_labels, color_labels):
    # TEST QUALITATIVE
    print("Mostrando pantalones negros")
    black_jeans = Retrieval_combined(imgs, class_labels, color_labels, "Jeans", "Black")
    ud.visualize_retrieval(black_jeans, len(black_jeans))
    #mostrar_imagenes(pantalones_negros)
    
    print("Mostrando ropa verde")
    ropa_verde = Retrieval_by_color(imgs[:100], color_labels,"Green")
    ud.visualize_retrieval(ropa_verde, len(ropa_verde))
    #mostrar_imagenes(ropa_verde)
    
    print("Mostrando vestidos")
    vestidos = Retrieval_by_shape(imgs[:100], class_labels,"Dresses")
    ud.visualize_retrieval(vestidos, len(vestidos))
    #mostrar_imagenes(vestidos)
    
if __name__ == '__main__':
    
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = ud.read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    
    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = ud.read_extended_dataset()
    cropped_images = ud.crop_images(imgs, upper, lower)
    
    
    # INICIALITZACIÓ KMEANS
    
    color_results = []
    Kmeans = []
    for image in test_imgs[:10]:
        km = KMeans(image,7)
        km.find_bestK(10,'FISHER')
        km.fit()
        Kmeans.append(km)

    for Kmean in Kmeans:
        ay = ud.visualize_k_means(Kmean, [80,60,3])
        print(Kmean.K)
    
    
    #test_qualitatiu(class_labels, color_labels)

    """
    imgs = imgs[:100]
    knn = KNN(imgs, class_labels)
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
        ud.visualize_k_means(Kmean, (335232,100))
    """
    WCD, time, iters = Kmean_statistics(Kmeans, 7)
    for wcd, times in zip(WCD, time):
        print(wcd, times)
    print(iters)
    
    # You can start coding your functions here