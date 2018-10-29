
# coding: utf-8

# In[ ]:


#Reference: Multi-Class Classification Tutorial with the Keras Deep Learning Library By Jason Brownlee 
#This is a Multi-Class Classification Tutorial with Keras Libray
#The dataset used in this example is the digits dataset


# In[1]:


#import necessery packages
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from pandas import read_csv
np.set_printoptions(threshold=np.nan)


# In[2]:


#Shuffling data and initializing coefficients with random values 
#use pseudorandom number generators. These little programs are often a function 
#that you can call that will return a random number. 
#This is important to ensure that the results we achieve from this model can be achieved again precisely. 
#It ensures that the stochastic process of training a neural network model can be reproduced
# fix random seed for reproducibility
#We can think of the seed as a parameter that determines the sequence of generated pseudorandom numbers.
seed = 7
np.random.seed(seed)


# In[3]:


# load dataset
#dataframe = read_csv('iris.csv', header=0, index_col=0, squeeze=True, encoding='latin-1')
dataframe = read_csv('digits.csv', header=0, index_col=None)
labels = read_csv('labels.csv', header=0, index_col=None)
X = dataframe.values
Y = labels.values


# In[4]:


print("features",dataframe)


# In[5]:


print("features size",dataframe.shape)


# In[6]:


#  One hot encoding of the output variable:
#We will reshape the output attribute from a vector that contains values for each class value to 
#be a matrix with a boolean for each class value
# LabelEncoder(): Encode labels with value between 0 and n_classes-1.
encoder = LabelEncoder()
#fit(y):Fit label encoder
encoder.fit(Y)
#transform(Y): Transform labels to normalized encoding.
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[7]:


print(dummy_y)


# In[8]:


#Below is a function that will create a baseline neural network for the iris classification problem. It creates a simple fully connected network with one hidden layer that contains 8 neurons.
#The network topology of this simple one-layer neural network can be summarized as:
#4 inputs -> [8 hidden nodes] -> 3 outputs
#a “softmax” activation function in the output layer is used to ensure the output values are in the range of 0 and 1 and may be used as predicted probabilities
# We will use the efficient Adam gradient descent optimization algorithm with a logarithmic loss function, which is called “categorical_crossentropy” in Keras

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=400, activation='relu'))
	model.add(Dense(8, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[9]:


#The Keras library provides wrapper classes to allow you to use neural network models developed with Keras in scikit-learn.
#There is a KerasClassifier class in Keras that can be used as an Estimator in scikit-learn, the base type of model in the library.
#The KerasClassifier takes the name of a function as an argument. This function must return the constructed neural network model, ready for training

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5)


# In[10]:


print(estimator)


# In[11]:


#Evaluate The Model with k-Fold Cross Validation
#we can define the model evaluation procedure. Here, we set the number of folds to be 10 (an excellent default) and to shuffle the data before partitioning it.

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

#Now we can evaluate our model (estimator) on our dataset (X and dummy_y) using a 10-fold cross-validation procedure (kfold).

results = cross_val_score(estimator, X, dummy_y, cv=kfold)


# In[12]:


print("Baseline: %.2f%% " % (results.mean()*100))

