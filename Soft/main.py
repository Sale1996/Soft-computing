import cv2
import utils
#biblioteka uz pomoc koje ucitavamo minst dataset
import tensorflow as tf
#za ispis
import matplotlib.pyplot as plt
import numpy as np
from keras import models


#ucitavanje podataka koji ce biti trening i test skup za neuronsku mrezu
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


#Ukoliko nema obucene neuronske mreze, ova sekcija koda bi se trebala izvrsiti
'''
ulazi_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_train)
izlazi_iz_neuronske = utils.pripremi_izlaz_za_neurosnku_mrezu(y_train)
ann = utils.kreiraj_neuronsku()
ann = utils.obuci_neuronsku(ann, ulazi_u_neuronsku, izlazi_iz_neuronske)
'''



#testiranje da li je dobro ucitao podatke, broj na 7777 treba da bude 8
'''
plt.imshow(x_test[4306], cmap='Greys')
plt.show()
'''

#Ucitavamo postojecu neuronsku mrezu
ann = models.load_model('obucenaNeuronska1.h5')

ulazi_za_test_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_test[4305: 4330])

rezultati = ann.predict(np.array(ulazi_za_test_u_neuronsku, np.float32))

print(utils.prikazi_rezultate(rezultati))
print(y_test[4305: 4330])

