__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

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
        for j in range(2, Kmax):
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
    WCD, ICD, FISHER, time, iters, k = Kmean_statistics(Kmeans, 7)
    fig1, axs1 = plt.subplots(1)
    fig2, axs2 = plt.subplots(1)
    fig3, axs3 = plt.subplots(1)
    fig4, axs4 = plt.subplots(1)
    fig5, axs5 = plt.subplots(1)
    fig6, axs6 = plt.subplots(1)
    fig7, axs7 = plt.subplots(1)
    fig8, axs8 = plt.subplots(1)
    fig9, axs9 = plt.subplots(1)
    
    axs1.set_title("WCD Kmeans")
    axs1.set_xlabel("Clusters")
    axs1.set_ylabel("Iterations")
    axs2.set_title("WCD Kmeans")
    axs2.set_xlabel("Clusters")
    axs2.set_ylabel("WCD")
    axs3.set_title("WCD Kmeans")
    axs3.set_xlabel("WCD")
    axs3.set_ylabel("Iterations")
    axs4.set_title("ICD Kmeans")
    axs4.set_xlabel("Clusters")
    axs4.set_ylabel("Iterations")
    axs5.set_title("ICD Kmeans")
    axs5.set_xlabel("Clusters")
    axs5.set_ylabel("ICD")
    axs6.set_title("ICD Kmeans")
    axs6.set_xlabel("ICD")
    axs6.set_ylabel("Iterations")
    axs7.set_title("FISHER Kmeans")
    axs7.set_xlabel("Clusters")
    axs7.set_ylabel("Iterations")
    axs8.set_title("FISHER Kmeans")
    axs8.set_xlabel("Clusters")
    axs8.set_ylabel("FISHER")
    axs9.set_title("FISHER Kmeans")
    axs9.set_xlabel("FISHER")
    axs9.set_ylabel("Iterations")
    
    media_WCD = [0 for i in range(len(WCD[0]))]
    media_ICD = [0 for i in range(len(WCD[0]))]
    media_FISHER = [0 for i in range(len(WCD[0]))]
    media_iters = [0 for i in range(len(WCD[0]))]
    media_clusters = [0 for i in range(len(WCD[0]))]
    
    for wcd, icd, fisher, times, iteration, clusters in zip(WCD, ICD, FISHER, time, iters, k):
        axs1.plot(clusters, iteration)
        axs2.plot(clusters, wcd)
        axs3.plot(wcd, iteration)
        axs4.plot(clusters, iteration)
        axs5.plot(clusters, icd)
        axs6.plot(icd, iteration)
        axs7.plot(clusters, iteration)
        axs8.plot(clusters, fisher)
        axs9.plot(fisher, iteration)
        i = 0
        for i in range(len(wcd)):
            media_WCD[i] += wcd[i]
            media_ICD[i] += icd[i]
            media_FISHER[i] += fisher[i]
            media_iters[i] += iteration[i]
            media_clusters[i] += clusters[i]
    
    plt.show()
    
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
    fig3, axs3 = plt.subplots(1)
    fig4, axs4 = plt.subplots(1)
    fig5, axs5 = plt.subplots(1)
    fig6, axs6 = plt.subplots(1)
    fig7, axs7 = plt.subplots(1)
    fig8, axs8 = plt.subplots(1)
    fig9, axs9 = plt.subplots(1)
    
    axs1.set_title("WCD Kmeans")
    axs1.set_xlabel("Clusters")
    axs1.set_ylabel("Iterations")
    axs2.set_title("WCD Kmeans")
    axs2.set_xlabel("Clusters")
    axs2.set_ylabel("WCD")
    axs3.set_title("WCD Kmeans")
    axs3.set_xlabel("WCD")
    axs3.set_ylabel("Iterations")
    axs4.set_title("ICD Kmeans")
    axs4.set_xlabel("Clusters")
    axs4.set_ylabel("Iterations")
    axs5.set_title("ICD Kmeans")
    axs5.set_xlabel("Clusters")
    axs5.set_ylabel("ICD")
    axs6.set_title("ICD Kmeans")
    axs6.set_xlabel("ICD")
    axs6.set_ylabel("Iterations")
    axs7.set_title("FISHER Kmeans")
    axs7.set_xlabel("Clusters")
    axs7.set_ylabel("Iterations")
    axs8.set_title("FISHER Kmeans")
    axs8.set_xlabel("Clusters")
    axs8.set_ylabel("FISHER")
    axs9.set_title("FISHER Kmeans")
    axs9.set_xlabel("FISHER")
    axs9.set_ylabel("Iterations")
    
    axs1.plot(clusters, iteration)
    axs2.plot(clusters, wcd)
    axs3.plot(wcd, iteration)
    axs4.plot(clusters, iteration)
    axs5.plot(clusters, icd)
    axs6.plot(icd, iteration)
    axs7.plot(clusters, iteration)
    axs8.plot(clusters, fisher)
    axs9.plot(fisher, iteration)
    
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
        km.find_bestK(10)
        km.fit()
        Kmeans.append(km)

    for Kmean in Kmeans:
        ay = ud.visualize_k_means(Kmean, [80,60,3])
        print(Kmean.K)
    
    #test_qualitatiu(class_labels, color_labels)
    test_quantitatiu(Kmeans)
    #knn = KNN(train_imgs, test_class_labels)
    #knn_labels = knn.predict(test_imgs, 3)
    
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
    