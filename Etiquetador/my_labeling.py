__authors__ = ['1639484', '1636492', '1638248']
__group__ = 'DJ.12'

import utils_data as ud
import numpy as np
from PIL import Image
from Kmeans import *
from KNN import *
import time as t
import matplotlib.pyplot as plt
import random

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
    ICD = []
    FISHER = []
    iters = []
    time = []
    clusters = []
    for i in range(len(Kmeans_list)):
        WCD_list = []
        ICD_list =[]
        FISHER_list = []
        time_list = []
        iters_list = []
        k = []
        Kmeans = Kmeans_list[i]
        for j in range(2, Kmax + 1):
            Kmeans.K = j
            inici = t.process_time()
            Kmeans.fit()
            final = t.process_time()
            temps = final - inici 
            time_list.append(temps)
            Kmeans.withinClassDistance()
            
            Kmeans.interClassDistance()
            Kmeans.fisher()
            WCD_list.append(Kmeans.WCD)
            ICD_list.append(Kmeans.ICD)
            FISHER_list.append(Kmeans.FISHER)
            iters_list.append(Kmeans.num_iter)
            k.append(j)
        WCD.append(WCD_list)
        ICD.append(ICD_list)
        FISHER.append(FISHER_list)
        time.append(time_list)
        iters.append(iters_list)
        clusters.append(k)
        
    return WCD, ICD, FISHER, time, iters, clusters


def get_shape_accuracy(classes, gt):
    ret = 0
    for clase in classes:
        if clase in gt:
            ret += 1
    return ret / len(classes)


def get_color_accuracy(colors, gt):
    ret = 0
    for i in range(len(colors)):
        if colors[i] == gt[i]:
            ret += 1
    return ret / len(classes)

#INSPIRACION
"""
def get_shape_accuracy(knn_labels, shape_labels):

    accuracy = 100*sum(1 for x, y in zip(sorted(knn_labels), sorted(shape_labels)) if x == y) / len(knn_labels)
    print(accuracy)


def get_color_accuracy(kmeans_labels, color_labels):
    accuracy = 100 * sum(1 for x, y in zip(sorted(kmeans_labels), sorted(color_labels)) if x == y) / len(kmeans_labels)
    print(accuracy)
"""
""" FI ANALISI QUANTITATIU """


def test_qualitatiu(class_labels, color_labels):
    # TEST QUALITATIVE
    print("Mostrando pantalones negros")
    black_jeans = Retrieval_combined(imgs, class_labels, color_labels, ["Jeans"], ["Black"])
    ud.visualize_retrieval(black_jeans, len(black_jeans))
    #mostrar_imagenes(pantalones_negros)
    
    print("Mostrando ropa verde")
    ropa_verde = Retrieval_by_color(imgs[:100], color_labels, ["Green"])
    ud.visualize_retrieval(ropa_verde, len(ropa_verde))
    #mostrar_imagenes(ropa_verde)
    
    print("Mostrando vestidos")
    vestidos = Retrieval_by_shape(imgs[:100], class_labels, ["Dresses"])
    ud.visualize_retrieval(vestidos, len(vestidos))
    #mostrar_imagenes(vestidos)


def test_quantitatiu(Kmeans):
    mostrar_K_statisticd(Kmeans)


def mostrar_K_statisticd(Kmeans):
    WCD, ICD, FISHER, time, iters, k = Kmean_statistics(Kmeans, 10)
    
    media_WCD = [0 for i in range(len(WCD[0]))]
    media_ICD = [0 for i in range(len(WCD[0]))]
    media_FISHER = [0 for i in range(len(WCD[0]))]
    media_iters = [0 for i in range(len(WCD[0]))]
    media_clusters = [0 for i in range(len(WCD[0]))]
    
    for wcd, icd, fisher, times, iteration, clusters in zip(WCD, ICD, FISHER, time, iters, k):
        for i in range(len(wcd)):
            media_WCD[i] += wcd[i]
            media_ICD[i] += icd[i]
            media_FISHER[i] += fisher[i]
            media_iters[i] += iteration[i]
            media_clusters[i] += clusters[i]
    
    for i in range(len(media_WCD)):
        media_WCD[i] /= len(WCD[i])
        media_ICD[i] /= len(ICD[i])
        media_FISHER[i] /= len(FISHER[i])
        media_iters[i] /= len(iters[i])
        media_clusters[i] /= len(k[i])
        
    mostrar_medias(media_WCD, media_ICD, media_FISHER, media_iters, media_clusters)


def mostrar_medias(wcd, icd, fisher, iteration, clusters):
    fig1, axs1 = plt.subplots(1)
    fig2, axs2 = plt.subplots(1)
    fig5, axs5 = plt.subplots(1)
    fig8, axs8 = plt.subplots(1)
    
    axs1.set_title("Kmeans")
    axs1.set_xlabel("Clusters")
    axs1.set_ylabel("Iterations")
    axs2.set_title("WCD Kmeans")
    axs2.set_xlabel("Clusters")
    axs2.set_ylabel("WCD")
    axs5.set_title("ICD Kmeans")
    axs5.set_xlabel("Clusters")
    axs5.set_ylabel("ICD")
    axs8.set_title("FISHER Kmeans")
    axs8.set_xlabel("Clusters")
    axs8.set_ylabel("FISHER")
    
    axs1.plot(clusters, iteration)
    axs2.plot(clusters, wcd)
    axs5.plot(clusters, icd)
    axs8.plot(clusters, fisher)
    
    plt.show()


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
    Kmeans = []
    for image in test_imgs[:10]:
        km = KMeans(image)
        km.find_bestK(10, 'FISHER')
        km.fit()
        Kmeans.append(km)
    
    """
    # VISUALIZAR KMEANS
    for Kmean in Kmeans:
        ay = ud.visualize_k_means(Kmean, [80,60,3])
        print(Kmean.K)
    
    # TESTS QUALITATIUS
    test_qualitatiu(class_labels, color_labels)
    """
    
    # TEST QUANTITATIU
    test_quantitatiu(Kmeans)

    #INICIALITZACIÓ KNN
    knn = KNN(train_imgs, train_class_labels)
    knn_labels = knn.predict(test_imgs, 4)
    print("Inicializado KNN")
    """
    #Quantitative functions
    n_images_s = 150
    total_trobats = []
    for it in range(0, 125):
            ti = random.randrange(10, 100)
            knn = KNN(train_imgs[:ti], train_class_labels[:ti])
            preds = knn.predict(test_imgs[:n_images_s], 4)
            
            trobats = Retrieval_by_shape(test_imgs[:n_images_s], preds, ["Dresses"])
            print(get_shape_accuracy(trobats, ["Dresses"]))
            ud.visualize_retrieval(trobats, len(trobats))
            total_trobats.extend(trobats)
    
    #get_shape_accuracy(total_trobats, test_class_labels)
    """
    