import cv2
import utils
import klase
#biblioteka uz pomoc koje ucitavamo minst dataset
import tensorflow as tf
#za ispis
import matplotlib.pyplot as plt
import numpy as np
from keras import models






'''

    OBUCAVANJE NEURONSKE MREZE

'''
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

#Ucitavamo postojecu neuronsku mrezu
ann = models.load_model('obucenaNeuronska1.h5')

#testiranje neuronske mreze
'''
ulazi_za_test_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_test[4305: 4330])
rezultati = ann.predict(np.array(ulazi_za_test_u_neuronsku, np.float32))
print(utils.prikazi_rezultate(rezultati))
print(y_test[4305: 4330])
'''








#ucitan video snimak
capture = cv2.VideoCapture('C:\\Users\\Admin\\Desktop\\soft projekat\\Soft-computing\\Soft\\videos\\video-0.avi')


broj_frejma = 0

#preuzimanje pocetnog frejma
capture.set(1, broj_frejma)
ret_val, frame = capture.read()


plt.imshow(frame)
plt.show()


'''

    UCITAVANJE PLAVE (SABIRANJE) I ZELENE (ODUZIMANJE) LINIJE
    
'''

binarna_slika_zelene_linije = utils.pronadji_zelenu_liniju(frame)
binarna_slika_plave_linije = utils.pronadji_plavu_liniju(frame)

zelena_linija = klase.Linija(utils.pronadji_temena_hog(binarna_slika_zelene_linije))
plava_linija = klase.Linija(utils.pronadji_temena_hog(binarna_slika_plave_linije))



#iscrtavanje pronadjenih linija koje je hog otkrio, radi provere
slika = utils.konvertuj_sliku_u_sivu(frame)

cv2.line(slika, (zelena_linija._Linija__prva_tacka[0], zelena_linija._Linija__prva_tacka[1]), (zelena_linija._Linija__druga_tacka[0], zelena_linija._Linija__druga_tacka[1]), (0, 255, 0), 3)
cv2.line(slika, (plava_linija._Linija__prva_tacka[0], plava_linija._Linija__prva_tacka[1]), (plava_linija._Linija__druga_tacka[0], plava_linija._Linija__druga_tacka[1]), (0, 255, 0), 3)

cv2.imshow("Slika", slika)
cv2.waitKey(0)
cv2.destroyAllWindows()




'''

    PREPOZNAVANJE BROJEVA SA SLIKE I KREIRANJE LISTE OBJEKATA BROJA

'''

capture.set(1, 1000)
povr_vred, test_brojevi_slika = capture.read()

slike_brojeva_sa_frejma_i_kordinate = utils.pronadji_brojeve(test_brojevi_slika)

