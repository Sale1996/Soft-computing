import cv2
import utils
#biblioteka uz pomoc koje ucitavamo minst dataset
import tensorflow as tf
#za ispis
import matplotlib.pyplot as plt
import numpy as np


#ucitavanje podataka koji ce biti trening i test skup za neuronsku mrezu
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#testiranje da li je dobro ucitao podatke, broj na 7777 treba da bude 8
#plt.imshow(x_test[200], cmap='Greys')
#plt.show()


ulazi_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_train[1:11])
izlazi_iz_neuronske = utils.pripremi_izlaz_za_neurosnku_mrezu(y_train[1:11])

ann = utils.kreiraj_neuronsku()
ann = utils.obuci_neuronsku(ann, ulazi_u_neuronsku, izlazi_iz_neuronske)

ulazi_za_test_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_test[200: 205])

rezultati = ann.predict(np.array(ulazi_za_test_u_neuronsku, np.float32))

print(utils.prikazi_rezultate(rezultati))
print(y_test[200:205])

