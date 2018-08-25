import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern

np.random.seed(2404)
import warnings
warnings.filterwarnings("ignore") 


import pickle
pickle_X = open("embed.pickle","rb")
X_full = pickle.load(pickle_X)
X_full = np.array(X_full)

pickle_Y = open("Y.pickle","rb+")
Y_full = pickle.load(pickle_Y)
Y_full = np.array(Y_full)


data = pd.DataFrame(X_full)
data['y'] = Y_full
data = data.sample(frac=1)
X = np.array(data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
Y = np.array(data['y'])



def GPClassify(X_train, Y_train, X_pred):
	gmm = GaussianProcessClassifier(kernel= 1*RBF(1) , multi_class='one_vs_rest' ,random_state=2404, n_restarts_optimizer=5)
	gmm.fit(X_train, Y_train)
	prob = gmm.predict_proba(X_pred)
	return prob

def GPPredict(X_train, Y_train, X_test, Y_test):
    gmm = GaussianProcessClassifier(kernel= 1*RBF(1) ,multi_class='one_vs_one', random_state=2404, n_restarts_optimizer=5)
    gmm.fit(X_train, Y_train)
    pred = gmm.predict(X_test)
    return np.mean(pred == Y_test)


def get_Acquisition_Uncertinity(X_train, Y_train, X_Sample):
	prob = GPClassify(X_train, Y_train, X_Sample)
	information = 1- np.max(prob, axis=1) # More means more uncertainity hence more information
	return information


def get_Acquisition_Similarity_Weighted_uncertinity(X_train, Y_train, X_Sample, AverageSimilarity ):
    information = get_Acquisition_Uncertinity(X_train, Y_train, X_Sample)
    weighted_information = np.multiply(information, AverageSimilarity)
    return weighted_information

def select_datapoint(X_train, Y_train, X_Sample, AverageSimilarity):
    info = get_Acquisition_Similarity_Weighted_uncertinity(X_train, Y_train, X_Sample, AverageSimilarity)
    return np.argmax(info)

def get_Average_similarity(X):
    X_rand = X[np.random.randint(1, len(X), size=5000 )]
    rbf  = RBF(1)
    M = rbf.__call__(X, X_rand)
    AverageSimilarity = np.mean(M, axis=1)
    return AverageSimilarity


AverageSimilarity = get_Average_similarity(X)


def get_Relative_Average_similarity(X, AverageSimilarity):
    X_rand = X[np.random.randint(1, len(X), size=5000 )]
    rbf  = RBF(1)
    M = rbf.__call__(X_rand, X)
    A_temp = 1/AverageSimilarity
    return np.mean(M*A_temp , axis=0)

Relative_AverageSimilarity = get_Relative_Average_similarity(X, AverageSimilarity)


def ActiveLearning(X_init, Y_init, X_Sample, Y_Sample, numSamples, AverageSimilarity):
    Precisions = np.zeros(numSamples + 2)
    index = select_datapoint(X_init, Y_init, X_Sample, AverageSimilarity)
    X_star = X_Sample[index]
    Y_star = Y_Sample[index]
    AverageSimilarity[index] = -1* AverageSimilarity[index]
    Pre = GPPredict(X_init, Y_init, X_Sample, Y_Sample)
    Precisions[0] = Pre
    print('Accuracy \t = \t' , Pre)
    for i in range(0, numSamples):
        X_init = np.vstack([X_init, X_star])
        Y_init = np.hstack([Y_init, Y_star])
        if((i+2)%5 ==0):
            Pre = GPPredict(X_init, Y_init, X_Sample, Y_Sample)
            print('Number of Labelled Data =',i + 2, '\t Accuracy = \t' , Pre)
        index = select_datapoint(X_init, Y_init, X_Sample, AverageSimilarity)
        AverageSimilarity[index] = -1* AverageSimilarity[index]
        X_star = X_Sample[index]
        Y_star = Y_Sample[index]

X_init = np.array(X[0:20,])
Y_init  = Y[0:20]
Relative_AverageSimilarity[0] = -1* Relative_AverageSimilarity[0]
Relative_AverageSimilarity[1] = -1* Relative_AverageSimilarity[1]
ActiveLearning(X_init, Y_init, X, Y, 100, Relative_AverageSimilarity)




















