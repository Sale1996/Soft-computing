import cv2
import klase

#biblioteka uz pomoc koje ucitavamo minst dataset
import tensorflow as tf
#za ispis
import matplotlib.pyplot as plt

import numpy as np

#Biblioteke vezane za neuronsku mrezu
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Activation

import random


#Omogucava nam euklitsko rastojanje da nadjemo
from scipy.spatial import distance

from PIL import Image



def skaliraj_sliku(slika):
    #Elementi matrice ciji su elementi 0 ili 255 treba skalirati na vrednosti 0 ili 1
    #Radi se normalizacija
    return slika/255



def pripremi_ulaz_za_neuronsku_mrezu(slike):
    #Funkcija koja skalira i pretvara element svakog buduceg ulaza u neuronsku mrezu
    ulazi_u_neuronsku = []
    for slika in slike:
        obradjena_slika = skaliraj_sliku(slika)
        ulazi_u_neuronsku.append(obradjena_slika)

    return ulazi_u_neuronsku

def pripremi_izlaz_za_neurosnku_mrezu(labele):

    # Funkcija koja ce na osnovu vrednosti formirati niz sacinjen od nula i jedne jedinice
    # Primera radi za broj 3 => 0001000000

    print("Priprema ulaz na neuronsku mrezu")
    zeljeni_izlazi = []
    for labela in labele:
        izlaz = np.zeros(10)
        izlaz[labela] = 1
        zeljeni_izlazi.append(izlaz)



    return np.array(zeljeni_izlazi)


def kreiraj_neuronsku():
    # kreirace se neuronska mreza sa 784 ulaza, 10 izlaza i sa 3 medjusloja od 256 ulaza
    print("Kreira neuronsku")
    model = Sequential()
    model.add(Dense(256, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def obuci_neuronsku(ann, x_train, y_train):
    # Obucavanje neuronske mreze

    # prebacivanje podataka u float32
    print("Obucava neuronsku")
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)

    #definisemo parametre za obucavanje
    ann.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    print("Krece obucavanje")
    #konkretno obucavanje neurosnke mreze
    #napomena: mensanje je stavljeno na false, posto su test skupovi inicijalno vec promesani
    ann.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1, shuffle=False)

    ann.save('obucenaNeuronska2.h5')


    return ann


def pobednik(izlaz_neuronske):
    # vraca indeks najvise pobudjenog clana izlaza sa neurosnke mreze

    return max(enumerate(izlaz_neuronske), key=lambda x: x[1])[0]


def prikazi_rezultate(izlazi):
    #na osnovu indeksa pobednika vracamo decimalni broj iz niza alfabeta
    alfabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    rezultati = []
    for izlaz in izlazi:
        rezultati.append(alfabet[pobednik(izlaz)])

    return rezultati



#rad sa frejmom

def konvertuj_sliku_u_sivu(slika):
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    slika = 255 - slika

    return cv2.cvtColor(slika, cv2.COLOR_RGB2GRAY)

def pronadji_plavu_liniju(slika):

    hsv = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
    # radimo dilaciju i eroziju kako bi izdvojili liniju
    slika_plava = cv2.inRange(hsv, (100, 25, 25), (140, 255, 255))
    slika_plava = 255 - slika_plava

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    slika_plava = cv2.erode(slika_plava, kernel, iterations=1)



    return slika_plava

def pronadji_zelenu_liniju(slika):

    hsv = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
    #radimo dilaciju i eroziju kako bi izdvojili liniju
    slika_zelena = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    slika_zelena= 255 - slika_zelena

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image_bin = cv2.dilate(slika_zelena, kernel, iterations=1)


    return image_bin


def pronadji_temena_hog(slika):

    img_bin = cv2.threshold(slika, 1, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(img_bin, 100, 250)
    img_blur = cv2.GaussianBlur(edges, (7, 7), 1)

    lines = cv2.HoughLinesP(img_blur, 1, np.pi / 180, 50, 50, 150)
    #umbro
    #sada smo nasli linije i trazimo onu najduzu, tj nasu liniju...

    nasa_linija = lines[0]


    '''
        pronalazenje min tacke po y
        i pronalazenje max tacke po y
    '''

    kordinate = nasa_linija[0]

    minY = (kordinate[0], kordinate[1])
    maxY = (kordinate[0], kordinate[1])

    #trazenje min Y
    for line in lines:
        coords = line[0]
        y1 = (coords[0], coords[1])
        y2 = (coords[2], coords[3])

        if(y1[1]<y2[1]):
            y=y1
        else:
            y=y2

        if y[1] <= minY[1]:
            minY = y

    #trazenje max Y
    for line in lines:
        coords = line[0]
        y1 = (coords[0], coords[1])
        y2 = (coords[2], coords[3])

        if (y1[1] > y2[1]):
            y = y1
        else:
            y = y2

        if y[1] >= maxY[1]:
            maxY = y


    kordinate = [minY[0],minY[1],maxY[0],maxY[1]]

    return kordinate


#Funkcija sa vezbi 2 koja pronalazi regione
def select_roi(image_orig, image_bin, zelena_linija, plava_linija):
    '''
        Funkcija sa vezbi
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        if h > 15 and h < 30 and w < 30:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom

            #GLEDAMO DA LI JE KONTURA OSAKACENA NEKOM LINIJOM... a to gledamo tako sto abs od razlika Y-a od centra konture i linije bude <10

            y_prave_za_x_tacke_zelena_linija = ((zelena_linija.druga_tacka[1] - zelena_linija.prva_tacka[1]) / (zelena_linija.druga_tacka[0] -zelena_linija.prva_tacka[0])) * (x -zelena_linija.prva_tacka[0]) +  zelena_linija.prva_tacka[1]
            y_prave_za_x_tacke_plava_linija = ((plava_linija.druga_tacka[1] -plava_linija.prva_tacka[1]) / (plava_linija.druga_tacka[0] - plava_linija.prva_tacka[0])) * ( x - plava_linija.prva_tacka[0]) + plava_linija.prva_tacka[1]

            if((abs(y+h/2 - y_prave_za_x_tacke_zelena_linija) < 15 and (x + w/2) > zelena_linija.druga_tacka[0] and (x + w/2)< zelena_linija.prva_tacka[0]) or (abs(y+h/2 -y_prave_za_x_tacke_plava_linija ) < 15 and (x + w/2)>plava_linija.druga_tacka[0] and (x + w/2)<plava_linija.prva_tacka[0])):
                print('')
            else:

                #da li je broj pojeden od strane nekog drugog ?


                prosiri_za = 25-w
                produzi_za = 25-h
                slika_broja_orig = image_orig[int(y-round(produzi_za/2,0) ) : int(y + h + round(produzi_za/2,0)) ,int(x - round(prosiri_za/2,0)) : int(x + w  + round(prosiri_za/2,0))]
                slika_broja = cv2.cvtColor(slika_broja_orig, cv2.COLOR_RGB2GRAY)
                slika_broja = cv2.threshold(slika_broja, 180, 255, cv2.THRESH_BINARY)[1]
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                slika_broja = cv2.dilate(slika_broja, kernel, iterations=2)
                #sada ako ga je preklopio neki broj onda ce ovde biti vise od 1 konture...
                img, konture, hierarchy = cv2.findContours(slika_broja, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                broj_kontura_jednog_broja = 0
               # plt.imshow(slika_broja)
               # plt.show()

                for kontura in konture:
                    xx, yy, ww, hh = cv2.boundingRect(kontura)
                    if(abs(13 - (xx+ww/2)) < 8 and abs(13-(yy+hh/2)) <8 and hh < 12  and ww < 12):
                        print('')
                    else:
                        broj_kontura_jednog_broja = broj_kontura_jednog_broja + 1


                if(broj_kontura_jednog_broja > 1):
                    print("")
                else:
                    region = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array.append([region, (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, regions_array

#vraca slicice brojeva sa njenim kordinatama
def pronadji_brojeve(slika, zelena_linija, plava_linija):


    img_bin = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_RGB2GRAY)
    img_bin = cv2.threshold(img_bin, 140, 255, cv2.THRESH_BINARY)[1]

    #brojevi su izdvojeni na slici
    img_bin = 255 - img_bin


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_bin = cv2.erode(img_bin, kernel, iterations=1)


    #dobijanje regiona
    origano, regioni = select_roi(slika, img_bin, zelena_linija, plava_linija)

    plt.imshow(origano)
    plt.show()

    brojevi_za_neuronsku = []

    for x in range(0, regioni.__len__()):

        brojevi_za_neuronsku.append([make_square(255-regioni[x][0]), regioni[x][1]])


    return brojevi_za_neuronsku



def make_square(im):
    '''

        Funkcija koja prima za parametar sliku i pretvara je u dimenzije 28x28
        ali tako da njenu originalnu velicinu ne dira, nego samo dodaje crne piksele sa strane
        kako bi popunio prazninu.

    '''
    desired_size = 28
    old_size = im.shape[:2]  # old_size is in (height, width) format
    new_size = tuple([int(x * 0.9) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im





def kreiraj_brojeve_trenutnog_frejma (slike_i_kordinate_frejma, ann):

    brojevi = []

    for x in range(0, slike_i_kordinate_frejma.__len__()):


        obradjena_slika = cv2.threshold(slike_i_kordinate_frejma[x][0], 5, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        obradjena_slika = cv2.erode(obradjena_slika, kernel, iterations=2)

        #plt.imshow(obradjena_slika, 'gray')
        #plt.show()

        #pronalazenje vrednosti cifre koja se nalazi na slici
        obradjena_slika = obradjena_slika.reshape(1, 784)
        obradjena_slika = pripremi_ulaz_za_neuronsku_mrezu(obradjena_slika)
        vrednost = prikazi_rezultate(ann.predict(np.array(obradjena_slika, np.float32)))

        #x = x + w/2
        x_kordinata_sredisnje_tacke = round(slike_i_kordinate_frejma[x][1][0] + slike_i_kordinate_frejma[x][1][2]/2)
        #y = y + h/2
        y_kordinata_sredisnje_tacke = round(slike_i_kordinate_frejma[x][1][1] + slike_i_kordinate_frejma[x][1][3]/2)

        broj = klase.Broj(vrednost, [x_kordinata_sredisnje_tacke, y_kordinata_sredisnje_tacke], False, False, [])

        brojevi.append(broj)


    return brojevi

def osvezi_rezultat (slika, lista_predhodnog_frejma, lista_trenutnog_frejma, zelena_linija, plava_linija, sirina_frejma, visina_frejma, rezultat , x):

    for broj_iz_trenutnog_frejma in lista_trenutnog_frejma:
        #za svaki broj trenutnog frejma gledamo da li je tek uleteo ili je postojao u predodnom frejmu...
        #ukoliko je postojao u predhodnom samo mu ubacujemo vrednosti kordianta tacaka, jer je to od kljucne vaznosti za dalji algoritam
        for broj_iz_predhodnog_frejma in lista_predhodnog_frejma:
            if(broj_iz_predhodnog_frejma.da_li_je_isti_broj(broj_iz_trenutnog_frejma)):
                broj_iz_trenutnog_frejma.kordinate_prve_tacke = broj_iz_predhodnog_frejma.kordinate_prve_tacke
                broj_iz_trenutnog_frejma.kordinate_druge_tacke = broj_iz_predhodnog_frejma.kordinate_druge_tacke



    for broj_iz_trenutnog_frejma in lista_trenutnog_frejma:
        #samo prve 2 iteracije ne gledamo da bude u granicama od 100 piksela sa x ili y strane!
        if(broj_iz_trenutnog_frejma.kordinate_prve_tacke[0] == -1 and (broj_iz_trenutnog_frejma.kordinate_sredisnje_tacke[0] <=100 or broj_iz_trenutnog_frejma.kordinate_sredisnje_tacke[1]<=100)): #or x < 150
            broj_iz_trenutnog_frejma.kordinate_prve_tacke = broj_iz_trenutnog_frejma.kordinate_sredisnje_tacke

        elif(broj_iz_trenutnog_frejma.kordinate_druge_tacke[0] == -1 and broj_iz_trenutnog_frejma.kordinate_prve_tacke[0] != -1):
            #sada upisuejmo kordinate i u ovaj atribut, pa zatim racunamo jednacinu prave i proveravamo da li sece neku od linija...
            broj_iz_trenutnog_frejma.kordinate_druge_tacke = broj_iz_trenutnog_frejma.kordinate_sredisnje_tacke

            if(plava_linija.da_li_ce_dotaci_liniju(broj_iz_trenutnog_frejma.kordinate_prve_tacke, broj_iz_trenutnog_frejma.kordinate_druge_tacke)):
                rezultat = rezultat + broj_iz_trenutnog_frejma.vrednost[0]
            #pa onda ispitujemo da li je preslo liniju za oduzimanje
            if(zelena_linija.da_li_ce_dotaci_liniju(broj_iz_trenutnog_frejma.kordinate_prve_tacke, broj_iz_trenutnog_frejma.kordinate_druge_tacke)):
                rezultat = rezultat - broj_iz_trenutnog_frejma.vrednost[0]



    return [lista_trenutnog_frejma,rezultat]
