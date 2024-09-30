#%%
# OK
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
###############################################################################
#               Toy dataset : 2 gaussians Ok
###############################################################################

n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.title('First data set')
plot_2d(X1, y1)

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print(clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset OK
###############################################################################

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# split train test
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

#%%
# Q1 Linear kernel Ok

# Pour la partie Q1 Linear kernel
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

# Utiliser GridSearchCV pour optimiser le modèle SVM avec noyau linéaire
clf_linear = GridSearchCV(SVC(), parameters, n_jobs=1)
clf_linear.fit(X_train, y_train)  # Entraînement du modèle

# Calculer les scores de généralisation et les afficher
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#%%
# Q2 polynomial kernel Ok

Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2) 
degrees = np.r_[1, 2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(SVC(), parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

print(clf_poly.best_params_)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


#%%
# display your results using frontiere Visualisation Ok 

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
###############################################################################
#               SVM GUI OK
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel


#%%
###############################################################################
#               Face Recognition Task OK
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

import numpy as np  # Ajoutez cette ligne pour corriger l'erreur
from sklearn.datasets import fetch_lfw_people

# Télécharger les données et les charger sous forme de tableaux numpy
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# Normalisation des images
images = images / 255.0  # Normaliser les valeurs des pixels entre 0 et 1

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()



#%%
####################################################################
# Extract features OK

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
#################################################################### OK
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q3 OK
import time 
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time.time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_train, y_train))

ind = np.argmax(scores)  # Index de la meilleure valeur de C
print("Best C: {}".format(Cs[ind]))

# Tracer les scores en fonction des valeurs de C
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Paramètres de régularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()

print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the training set")
t0 = time.time()

# Prédire avec le meilleur C trouvé
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time.time() - t0))
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_train, y_train))

#%%
# Réentraîner le classificateur avec le meilleur C trouvé
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)

# Prédire les étiquettes pour les images du jeu de test
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time.time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib OK

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


#%%
# Q4 OK

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)  # Exécution sur les données sans bruit

print("Score avec variable de nuisance")
n_features = X.shape[1]

# On rajoute des variables de nuisance
sigma = 1
noise = sigma * np.random.randn(n_samples, 250)
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy, y) 


#%%
# Q5 Ok mais très long 
n_components = 1  # jouer avec ce parametre
print(f'Score après réduction de dimension avec {n_components} composantes principales')
pca = PCA(n_components=n_components).fit(X_noisy)
X_pca = pca.transform(X_noisy)
run_svm_cv(X_pca,y)
 
# %%
