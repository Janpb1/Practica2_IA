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

def get_shape_accuracy(knn_labels, Ground_Truth):
    count = 0
    for predict, clase in zip(knn_labels, Ground_Truth):
        if predict == clase:
            count += 1
    return 100*(count / len(knn_labels))

def mostrar_shape_accuracy(knn):
    porcentajes = []
    K = []
    for k in range(3, 15):
        knn_labels = knn.predict(test_imgs, k)
        x = get_shape_accuracy(knn_labels, test_class_labels)
        porcentajes.append(x)
        K.append(k)
        knn.neighbors=[]
        print("Per a K = ", k, " el percentatge d'encerts és ",x)

    plt.title("KNN shape accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.plot(K, porcentajes)
    plt.show()


def get_color_accuracy(kmeans_labels, Ground_Truth):
    count = 0
    for km_label, gt_label in zip(kmeans_labels, Ground_Truth):
        if km_label == gt_label:
            count += 1
    return 100*(count / len(kmeans_labels))

""" FI ANALISI QUANTITATIU """

"""VISUALITZACIÓ ANALISIS QALITATIU"""
def test_qualitatiu(class_labels, color_labels):
    # TEST QUALITATIVE
    print("Mostrando pantalones negros")
    val = Retrieval_combined(imgs, class_labels, color_labels, ["Jeans"], ["Black"])
    ud.visualize_retrieval(val, len(val))
    
    print("Mostrando vestidos rosas y azules")
    val = Retrieval_combined(imgs, class_labels, color_labels, ["Dresses"], ["Pink", "Blue"])
    ud.visualize_retrieval(val, len(val))
    
    print("Mostrando vestidos y sandalias rosas ")
    val = Retrieval_combined(imgs, class_labels, color_labels, ["Dresses", "Sandals"], ["Pink"])
    ud.visualize_retrieval(val, len(val))
    
    print("Mostrando ropa rosa")
    val = Retrieval_by_color(imgs[:30], color_labels, ["Pink"])
    ud.visualize_retrieval(val, len(val))
    
    print("Mostrando ropa blanca y azul")
    val = Retrieval_by_color(imgs[:30], color_labels, ["White", "Blue"])
    ud.visualize_retrieval(val, len(val))
    
    print("Mostrando vaqueros")
    val = Retrieval_by_shape(imgs[:30], class_labels, ["Jeans"])
    ud.visualize_retrieval(val, len(val))
    
    print("Mostrando calcetines y bolsas")
    val = Retrieval_by_shape(imgs[:30], class_labels, ["Socks", "Handbags"])
    ud.visualize_retrieval(val, len(val))

"""VISUALITZACIÓ ANALISIS QUANTITATIU"""
def test_quantitatiu(Kmeans, knn):
    mostrar_K_statisticd(Kmeans)
    mostrar_shape_accuracy(knn)
    
    
if __name__ == '__main__':
    
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = ud.read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = ud.read_extended_dataset()
    cropped_images = ud.crop_images(imgs, upper, lower)

    heuristiques = ['WCD', 'ICD', 'FISHER']
    # INICIALITZACIÓ KMEANS
    for heuristica in heuristiques:
        Kmeans = []
        for image in test_imgs[:10]:
            km = KMeans(image)
            km.find_bestK(10, heuristica)
            km.fit()
            Kmeans.append(km)
        
        # VISUALIZAR KMEANS
        for Kmean in Kmeans:
            ay = ud.visualize_k_means(Kmean, [80,60,3])
            print(Kmean.K)
    
    # TESTS QUALITATIUS
    test_qualitatiu(class_labels, color_labels)
    
    #INICIALITZACIÓ KNN
    knn = KNN(train_imgs, train_class_labels)
    
    # TEST QUANTITATIU
    test_quantitatiu(Kmeans, knn)

    """Kmeans = []
    for image in test_imgs[:10]:
        km = KMeans(image)
        km.fit()
        Kmeans.append(km)
    kmeans_labels = []
    for i in range(len(Kmeans)):
        kmeans_labels.append(list(get_colors(Kmeans[i].centroids)))
    get_color_accuracy(kmeans_labels, test_color_labels[:10])"""