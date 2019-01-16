import cv2

#biblioteka uz pomoc koje ucitavamo minst dataset
import tensorflow as tf
#za ispis
import matplotlib.pyplot as plt

import numpy as np

#Biblioteke vezane za neuronsku mrezu
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD



def skaliraj_sliku(slika):
    #Elementi matrice ciji su elementi 0 ili 255 treba skalirati na vrednosti 0 ili 1
    #Radi se normalizacija
    return slika/255

def matrica_u_vektor(slika):
    #Sliku, odnosno matricu, dimenzija 28x28 pretvara u vektor sa 784 elementa
    return slika.flatten()

def pripremi_ulaz_za_neuronsku_mrezu(slike):
    #Funkcija koja skalira i pretvara element svakog buduceg ulaza u neuronsku mrezu
    ulazi_u_neuronsku = []
    for slika in slike:
        skalirana_slika = skaliraj_sliku(slika)
        ulazi_u_neuronsku.append(matrica_u_vektor(skalirana_slika))

    return ulazi_u_neuronsku

def pripremi_izlaz_za_neurosnku_mrezu(labele):

    # Funkcija koja ce na osnovu vrednosti formirati niz sacinjen od nula i jedne jedinice
    # Primera radi za broj 3 => 0001000000

    print("Priprema ulaz na neuronsku mrezu")
    zeljeni_izlazi = []
    for labela in labele:
        izlaz = np.zeros(10)
        izlaz[labela-1] = 1
        zeljeni_izlazi.append(izlaz)

    return np.array(zeljeni_izlazi)


def kreiraj_neuronsku():
    # kreirace se neuronska mreza sa 784 ulaza, 10 izlaza i sa medjuslojem od 128 ulaza
    print("Kreira neuronsku")
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann

def obuci_neuronsku(ann, x_train, y_train):
    # Obucavanje neuronske mreze

    # prebacivanje podataka u float32

    print("Obucava neuronsku")
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)

    #definisemo parametre za obucavanje
    sgd = SGD(lr=0.01, momentum = 0.9)
    ann.compile(loss='mean_squared_error',optimizer=sgd)

    #konkretno obucavanje neurosnke mreze
    #napomena: mensanje je stavljeno na false, posto su test skupovi inicijalno vec promesani
    ann.fit(x_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann


def pobednik(izlaz_neuronske):
    # vraca indeks najvise pobudjenog clana izlaza sa neurosnke mreze

    print("bira pobednika")
    return max(enumerate(izlaz_neuronske), key=lambda x: x[1])[0]


def prikazi_rezultate(izlazi):
    #na osnovu indeksa pobednika vracamo decimalni broj iz niza alfabeta
    alfabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    rezultati = []
    for izlaz in izlazi:
        rezultati.append(alfabet[pobednik(izlaz)])

    return rezultati