'''

    KLASA KOJA PREDSTAVLJA LINIJU, U OVOM SLUCAJU TO CE BITI PLAVA I ZELENA LINIJA

'''

class Linija:

    __prva_tacka = [0, 0]
    __druga_tacka = [0, 0]

    def __init__(self, kordinate_linije):

        self.__prva_tacka = (kordinate_linije[0], kordinate_linije[1])
        self.__druga_tacka = (kordinate_linije[2], kordinate_linije[3])






'''

    KLASA KOJA PREDSTAVLJA BROJ U SEBI IMA:
        VREDNOST,
        KORDINATE GORNJE LEVE TACKE,
        KORDINATE DONJE DESNE TACKE,
        BOOLEAN DA LI JE PRESLA LINIJU ZA SABIRANJE,
        BOOLEAN DA LI JE PRESLA LINIJU ZA ODUZIMANJE,
        LISTU BROJEVA KOJE JE PREKLOPIO

'''

from scipy.spatial import distance

class Broj :

    __vrednost = -1
    __kordinate_sredisnje_tacke= [-1,-1]
    __da_li_je_preslo_sabiranje = False
    __da_li_je_preslo_oduzimanje = False
    __preklopljeni_brojevi = []

    def __init__(self, vrednost, kordinate_sredisnje_tacke, preslo_sabiranje, preslo_oduzimanje, preklopljeni_brojevi):

        self.__vrednost = vrednost
        self.__kordinate_sredisnje_tacke = kordinate_sredisnje_tacke
        self.__da_li_je_preslo_sabiranje = preslo_sabiranje
        self.__da_li_je_preslo_oduzimanje = preslo_oduzimanje
        self.__preklopljeni_brojevi = preklopljeni_brojevi


    def da_li_je_isti_broj(self, broj):

        if(self.__vrednost == broj._Broj__vrednost):
            distanca =distance.euclidean(self.__kordinate_sredisnje_tacke, broj._Broj__kordinate_sredisnje_tacke)
            if(distanca < 100):
                return True
            else:
                return False
