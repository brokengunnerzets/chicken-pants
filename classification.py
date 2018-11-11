# Import des libraries necessaires
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
from sklearn.model_selection import train_test_split, KFold

import numpy as np
import os
from NeuralNetworkImp import NeuralNetwork


#--------------------------------------------------------------------------------------------------------
#
# -------------------------------------MAIN PROGRAM------------------------------------------------------

# On a besoin de cette fonction car la conversion de nparray laisse toutes les données en np_str et on veux float
def convertStringsToFloats(array):
    convertedArray = []
    for element in array:
        convertedArray.append(float(element))
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

#Exemple d'utilisation des graphiques à travers multiples hyper paramètres
def decisionClassifyWithTree(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures, case):

    #Paramètres du decision tree
    decisionTreeParams = [0, 3, 5, 10]
    accuracyScoreList = []
    print(case)
    print()

    # Creation de nos arbres de decision en utilisant l'entropie et un max_depth variable
    for element in decisionTreeParams:
        if element == 0:
            desTree = tree.DecisionTreeClassifier(criterion='entropy')
            element = "None"
        else:
            desTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=element)

        # Apprentissage avec la fonction fit
        desTree = desTree.fit(trainFeatures, trainLabels)       
        
        predictionLabels = desTree.predict(validationFeatures)
        
        # Accuracy Score
        print("Accuracy score : Arbe de décision max_depth = " + str(element))
        print(desTree.score(testFeatures, testLabels))
        accuracyScoreList.append(desTree.score(testFeatures, testLabels))

        
        if (isinstance(validationLabels[0], str) == False):
            validationLabels = arrangeLabels(validationLabels)
            
        if (isinstance(predictionLabels[0], str) == False):
            predictionLabels = arrangeLabels(predictionLabels)
        
        print("F1 score : Arbe de décision max_depth = " + str(element))
        print(f1_score(validationLabels, predictionLabels, average='weighted'))
        print('\n')
     
    #Accuracy Graphique
    x = decisionTreeParams
    y = accuracyScoreList

    plt.plot(x, y)

    plt.title('Comparaison de Hyper Param avec Arbre de décision')
    plt.xlabel('Primitive 1 - Hyper Param')
    plt.ylabel('Primitive 2 - Accuracy Score')
    plt.show()

# Pour la validation croisée
def crossValidationRN(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures, case):
    
    print()
    print(case)
    
    dataFeatures = trainFeatures + validationFeatures + testFeatures
    dataLabels = trainLabels + validationLabels + testLabels
    
    accuracyScoreList = []
    f1ScoreList = []
    
    kf = KFold(n_splits=3)
    for train_index, test_index in kf.split(dataFeatures):
        
        newTrainFeatures = []
        newTrainLabels = []
        newTestFeatures = []
        newTestLabels = []
        
        
        for index in train_index:
            newTrainFeatures.append(dataFeatures[index])
            newTrainLabels.append(dataLabels[index])
            
        for index in test_index:
            newTestFeatures.append(dataFeatures[index])
            newTestLabels.append(dataLabels[index])


        # #TODO Replace with usable code in RN fait maison!
        #
        # desTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
        #
        # # Apprentissage avec la fonction fit
        # desTree = desTree.fit(newTrainFeatures, newTrainLabels)
        # predictionLabels = desTree.predict(newTestFeatures)
        #
        # # Accuracy Score
        # accuracyScoreList.append(desTree.score(newTestFeatures, newTestLabels))
        #
        # if (isinstance(validationLabels[0], str) == False):
        #     validationLabels = arrangeLabels(validationLabels)
        #
        # if (isinstance(predictionLabels[0], str) == False):
        #     predictionLabels = arrangeLabels(predictionLabels)
        #
        # #f1 score
        # f1ScoreList.append(f1_score(newTestLabels, predictionLabels, average='weighted'))
    
    print("Cross-Val Accuracy: ")   
    print(np.mean(accuracyScoreList))
    print("Cross-Val F1: ")
    print(np.mean(f1ScoreList))
    
#KNN
def decisionWithKNN(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(trainFeatures, trainLabels)
    predictionlabels = neigh.predict(validationFeatures + testFeatures)
    print(predictionlabels)

    print("F1 score : Algorithme KNN avec valeur de K = 3" )
    print(f1_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print('\n')


#SVM
def decisionWithSVM(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]

    svc = SVC(gamma='auto')
    svc.fit(trainFeatures, trainLabels)
    predictionlabels = svc.predict(validationFeatures + testFeatures)
    print(predictionlabels)

    print("F1 score : Algorithme SVM")
    print(f1_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print('\n')


#RN
def decisionWithNN(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    predictionlabels=[]

    mlp = MLPClassifier()
    mlp.fit(trainFeatures, trainLabels)
    predictionlabels = mlp.predict(validationFeatures + testFeatures)
    print(predictionlabels)

    print("F1 score : Algorithme RN")
    print(f1_score(validationLabels+testLabels, predictionlabels, average='weighted'))
    print('\n')


#RN fait maison
def decisionWithNNScratch(trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures):

    # #TODO Correction of RN to work with train labels and train features
    # predictionlabels=[]
    #
    # nn =  NeuralNetwork(3, 4, 1)
    # nn.train_network(trainFeatures, trainLabels, 10)
    # predictionlabels = nn.forward_propagated()
    # print(predictionlabels)
    #
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
imagesFolder = "C:/Users/turco/PycharmProjects/chicken-pants/Ensemble_B/"

imageFeatureVector = []
labelVector =[]

print("Images Folder: ")
print(imagesFolder)

x = next(os.walk(imagesFolder))[1]
print("x: ")
print(x)

for element in x:
    for filename in os.listdir(imagesFolder + element):
        print(element)
        print(filename)
        print("/Users/turco/PycharmProjects/chicken-pants/Ensemble_B/" + element + "/" + filename)
        img1 = Image.open("/Users/turco/PycharmProjects/chicken-pants/Ensemble_B/" + element + "/" + filename)# Chemin a votre image
        img1 = img1.resize((320, 640), Image.ANTIALIAS)
        img1 = img1.convert('L')
        imageFeatureVector.append(list(img1.getdata()))
        labelVector.append([element])

print(len(imageFeatureVector))


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
#
#
#------------------Validation Croisée-----------------------
#
#
#
#


print()
print("----VALIDATION CROISÉE-----")
print()

#
#
# #--------------------------------------------------------------------------------------------------------
# #
# # -------------------------------------END PROGRAM-------------------------------------------------------