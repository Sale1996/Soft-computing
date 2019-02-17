import cv2
import utils
import klase
#biblioteka uz pomoc koje ucitavamo minst dataset
import tensorflow as tf
#za ispis
import matplotlib.pyplot as plt
import numpy as np
from keras import models

from scipy.spatial import distance




'''

    OBUCAVANJE NEURONSKE MREZE

'''
#ucitavanje podataka koji ce biti trening i test skup za neuronsku mrezu
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



x_train1= []
y_train1= []

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

for x in range(x_train.__len__()):
    x_train1.append(cv2.threshold(x_train[x], 170, 255, cv2.THRESH_BINARY)[1])

    x_train1.append(cv2.erode(x_train[x], kernel, iterations=1))
    x_train1.append(cv2.erode(x_train[x], kernel, iterations=1))


    x_train1.append(cv2.dilate(x_train[x], kernel, iterations=1))
    x_train1.append(cv2.dilate(x_train[x], kernel, iterations=2))


    y_train1.append(y_train[x])
    y_train1.append(y_train[x])
    y_train1.append(y_train[x])
    y_train1.append(y_train[x])
    y_train1.append(y_train[x])






for x in range(x_test.__len__()):
    x_test[x] = cv2.threshold(x_test[x],190, 255, cv2.THRESH_BINARY)[1]




x_train1 = np.asarray(x_train1).reshape(300000, 784)
x_test = x_test.reshape(10000, 784)


#Ukoliko nema obucene neuronske mreze, ova sekcija koda bi se trebala izvrsiti
'''
ulazi_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_train1)
izlazi_iz_neuronske = utils.pripremi_izlaz_za_neurosnku_mrezu(y_train1)
ann = utils.kreiraj_neuronsku()
ann = utils.obuci_neuronsku(ann, ulazi_u_neuronsku, izlazi_iz_neuronske)
'''

#Ucitavamo postojecu neuronsku mrezu
ann = models.load_model('obucenaNeuronska2.h5')

#testiranje neuronske mreze



'''
ulazi_za_test_u_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_test[4305: 4330])
rezultati = ann.predict(np.array(ulazi_za_test_u_neuronsku, np.float32))
print(utils.prikazi_rezultate(rezultati))
print(y_test[4305: 4330])


'''
'''
tacno = 0
netacno = 0

for x in range(1, 10000):
    ulaz_za_neuronsku = utils.pripremi_ulaz_za_neuronsku_mrezu(x_test[x: x+1])
    rezultat = ann.predict(np.array(ulaz_za_neuronsku, np.float32))
    if(utils.prikazi_rezultate(rezultat)[0] == y_test[x]):
        tacno = tacno + 1

    else:
        netacno = netacno + 1



print("Tacno je" + str(tacno))
print("Netacno je " + str(netacno))

'''





#ucitan video snimak
capture = cv2.VideoCapture('C:\\Users\\Admin\\Desktop\\soft projekat\\Soft-computing\\Soft\\videos\\video-2.avi')


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

cv2.line(slika, (zelena_linija.prva_tacka[0], zelena_linija.prva_tacka[1]), (zelena_linija.druga_tacka[0], zelena_linija.druga_tacka[1]), (0, 255, 0), 3)
cv2.line(slika, (plava_linija.prva_tacka[0], plava_linija.prva_tacka[1]), (plava_linija.druga_tacka[0], plava_linija.druga_tacka[1]), (0, 255, 0), 3)

cv2.imshow("Slika", slika)
cv2.waitKey(0)
cv2.destroyAllWindows()




'''

    PREPOZNAVANJE BROJEVA SA SLIKE I KREIRANJE LISTE OBJEKATA BROJA

'''
#uzimamo informacije o video snimku kao sto su broj frejmova, visina i sirina prozora...
videoLength = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH )
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT )
lista_brojeva_predhodnog_frejma = []
ista_brojeva_trenutnog_frejma = []

rezultat= 0

for x in range(1, videoLength, 5):

    capture.set(1, x)

    povr_vred, test_brojevi_slika = capture.read()

    slike_brojeva_sa_frejma_i_kordinate = utils.pronadji_brojeve(test_brojevi_slika, zelena_linija, plava_linija)

    lista_brojeva_trenutnog_frejma = utils.kreiraj_brojeve_trenutnog_frejma(slike_brojeva_sa_frejma_i_kordinate, ann)

    #ukoliko je prva iteracija onda postavljamo da je lista predhodnog frejma == nasoj trenutnoj
    if(x == 1):

        lista_brojeva_predhodnog_frejma = lista_brojeva_trenutnog_frejma

    else:
        #definisemo ostale atribute trenutnog frejma iiiiii nalazimo nestale brojeve

        lista_brojeva_trenutnog_frejma = utils.pronadji_nestale_brojeve_i_definisi_trenutne(test_brojevi_slika ,lista_brojeva_predhodnog_frejma, lista_brojeva_trenutnog_frejma, zelena_linija, plava_linija, width, height )

        lista_brojeva_trenutnog_frejma = utils.pronadji_novonastale_brojeve(lista_brojeva_predhodnog_frejma, lista_brojeva_trenutnog_frejma, zelena_linija, plava_linija, width, height )

        lista_brojeva_predhodnog_frejma = lista_brojeva_trenutnog_frejma

        rezultat = utils.azurirajRezultat(rezultat, zelena_linija,plava_linija,lista_brojeva_trenutnog_frejma)



print("Konacni rezultat je : " + str(rezultat))















