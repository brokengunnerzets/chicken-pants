# Import des libraries necessaires
import time

from PIL import Image
from matplotlib import pyplot as plt

#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
from sklearn import tree

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
from sklearn.svm import SVC

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPClassifier

#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve

#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
from sklearn.model_selection import train_test_split, KFold

import numpy as np
import os
from NeuralNetworkImpl import NeuralNetworkImpl


# --------------------------------------------------------------------------------------------------------
#
# -------------------------------------MAIN PROGRAM------------------------------------------------------

#On a besoin de cette fonction car la conversion de nparray laisse toutes les données en np_str et on veux float
def convertStringsToFloats(array):
    convertedArray = []
    for element in array:
        convertedArray.append(int(element))
    return convertedArray

#Séparation des données en test/train
def floatMapTrainValidTest(data, labels):

    trainFeatures, validFeatures, trainLabels, validLabels = train_test_split(data, labels, test_size=0.30)
    validFeatures, testFeatures, validLabels, testLabels = train_test_split(validFeatures, validLabels, test_size=0.33)
    
    floatDataMap = {"trainFeatures" : [], 
                   "trainLabels" : [], 
                   "validFeatures" : [],
                   "validLabels" : [],
                   "testFeatures" : [], 
                   "testLabels" : []
                   }
    
    for row in trainFeatures:
        floatDataMap["trainFeatures"].append(convertStringsToFloats(row))
        
    for row in validFeatures:
        floatDataMap["validFeatures"].append(convertStringsToFloats(row))
        
    for row in testFeatures:
        floatDataMap["testFeatures"].append(convertStringsToFloats(row))
        
    for row in trainLabels:
        floatDataMap["trainLabels"].append(row)
        
    for row in validLabels:
        floatDataMap["validLabels"].append(row)
        
    for row in testLabels:
        floatDataMap["testLabels"].append(row)
        
    return floatDataMap

#
#
#
#
#------------------Section Définition de fonctions-----------
#
#
#
#

#Conversion des labels de validation de format bizarres
def arrangeLabels(labels):
    newLabels = []
    for label in labels:
        newLabels.append(label[0])
    return newLabels

#KNN
def decisionWithKNN(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]
    neigh = KNeighborsClassifier(n_neighbors=3)

    start_time = time.time()
    neigh.fit(trainFeatures, trainLabels)
    print("KNN Train")
    print("----------%s Seconds-----" % (time.time() - start_time))

    start_time = time.time()
    predictionlabels = neigh.predict(validationFeatures + testFeatures)
    print("KNN Predict")
    print("----------%s Seconds-----" % (time.time() - start_time))

    print("F1 score : Algorithme KNN avec valeur de K = 3" )
    print(f1_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print("Precision score : Algorithme KNN avec valeur de K = 3")
    print(precision_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print("Recall score : Algorithme KNN avec valeur de K = 3")
    print(recall_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print('\n')

    print("Matrice de confusion - KNN")
    print(confusion_matrix(validationLabels+testLabels, predictionlabels))

#SVM
def decisionWithSVM(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]
    svc = SVC(gamma='auto')

    start_time = time.time()
    svc.fit(trainFeatures, trainLabels)
    print("SVM Train")
    print("----------%s Seconds-----" % (time.time() - start_time))

    start_time = time.time()
    predictionlabels = svc.predict(validationFeatures + testFeatures)
    print("SVM Predict")
    print("----------%s Seconds-----" % (time.time() - start_time))

    print("F1 score : Algorithme SVM")
    print(f1_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print("Precision score : Algorithme SVM")
    print(precision_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print("Recall score : Algorithme SVM")
    print(recall_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print('\n')

    print("Matrice de confusion - SVM")
    print(confusion_matrix(validationLabels+testLabels, predictionlabels))

#RN
def decisionWithNN(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]
    mlp = MLPClassifier()

    start_time = time.time()
    mlp.fit(trainFeatures, trainLabels)
    print("NN MLP Train")
    print("----------%s Seconds-----" % (time.time() - start_time))

    start_time = time.time()
    predictionlabels = mlp.predict(validationFeatures + testFeatures)
    print("NN MLP Predict")
    print("----------%s Seconds-----" % (time.time() - start_time))

    print("F1 score : Algorithme RN")
    print(f1_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print("Precision score : Algorithme RN")
    print(precision_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print("Recall score : Algorithme RN")
    print(recall_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print('\n')

    print("Matrice de confusion - NN")
    print(confusion_matrix(validationLabels+testLabels, predictionlabels))

#RN fait maison
def decisionWithNNScratch(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]

    start_time = time.time()

    trainLabels = np.asarray(trainLabels)/8
    trainFeatures = np.asarray(trainFeatures)/np.amax(np.asarray(trainFeatures), axis=0)

    nn = NeuralNetworkImpl(204800, 134, 1)
    nn.define_weights()
    nn.train_network(trainFeatures, trainLabels, 1)
    predictionlabels = nn.predict(trainFeatures)

    print("RN Train/Predict")
    print("----------%s Seconds-----" % (time.time() - start_time))

    print(predictionlabels)
    print(trainLabels)

    print(len(trainLabels))
    # print("F1 score : Algorithme RN")
    # print(f1_score(trainLabels, predictionlabels, average='weighted'))
    # print('\n')

#
#
#
#------------------Section Extraction d'images -----------
#
#
#
#

# À modifier pour votre environmement - ensemble B
imagesFolder = "C:/Users/maxi4/chicken-pants/Ensemble_B/"

imageFeatureVector = []
labelVector =[]

x = next(os.walk(imagesFolder))[1]
classNumber = 0

for element in x:
    classNumber = classNumber + 1
    for filename in os.listdir(imagesFolder + element):
        img1 = Image.open("/Users/maxi4/chicken-pants/Ensemble_B/" + element + "/" + filename)# Chemin a votre image
        img1 = img1.resize((320, 640), Image.ANTIALIAS)
        img1 = img1.convert('L')
        imageFeatureVector.append(list(img1.getdata()))
        labelVector.append([classNumber])

print(len(labelVector))
# Séparation de train/valid/test
imagesVector = {"trainFeatures" : [],
               "trainLabels" : [],
               "validFeatures" : [],
               "validLabels" : [],
               "testFeatures" : [],
               "testLabels" : []
               }

#Création du split hold-out
imagesMap = floatMapTrainValidTest(imageFeatureVector, labelVector)


#
#
#
#
#------------------Section Classifications---------------------
#
#
#
#

#Classification KNN
decisionWithKNN(imagesMap["trainLabels"], imagesMap["trainFeatures"],
                imagesMap["validLabels"], imagesMap["validFeatures"],
                imagesMap["testLabels"],imagesMap["testFeatures"])

#Classification SVM
decisionWithSVM(imagesMap["trainLabels"], imagesMap["trainFeatures"],
                imagesMap["validLabels"], imagesMap["validFeatures"],
                imagesMap["testLabels"],imagesMap["testFeatures"])

#Classification RN
decisionWithNN(imagesMap["trainLabels"], imagesMap["trainFeatures"],
                imagesMap["validLabels"], imagesMap["validFeatures"],
                imagesMap["testLabels"],imagesMap["testFeatures"])

#Classification RN fait maison
decisionWithNNScratch(imagesMap["trainLabels"], imagesMap["trainFeatures"],
                imagesMap["validLabels"], imagesMap["validFeatures"],
                imagesMap["testLabels"],imagesMap["testFeatures"])

#
#
# #--------------------------------------------------------------------------------------------------------
# #
# # -------------------------------------END PROGRAM-------------------------------------------------------