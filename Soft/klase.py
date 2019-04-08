'''

    KLASA KOJA PREDSTAVLJA LINIJU, U OVOM SLUCAJU TO CE BITI PLAVA I ZELENA LINIJA

'''

class Linija:

    prva_tacka = [0, 0]
    druga_tacka = [0, 0]

    def __init__(self, kordinate_linije):

        self.prva_tacka = (kordinate_linije[0], kordinate_linije[1])
        self.druga_tacka = (kordinate_linije[2], kordinate_linije[3])


    def da_li_ce_dotaci_liniju(self, druga_tacka, prva_tacka):


        for x in range(self.druga_tacka[0],self.prva_tacka[0]):
            if(x >= druga_tacka[0]):
                y_prave_za_x_tacke_linija = ((self.druga_tacka[1] - self.prva_tacka[1]) / (
                            self.druga_tacka[0] - self.prva_tacka[0])) * (x - self.prva_tacka[0]) + self.prva_tacka[1]

                y_prave_kretanja_broja = ((prva_tacka[1] - druga_tacka[1]) / (
                        prva_tacka[0] - druga_tacka[0])) * (x - druga_tacka[0]) + druga_tacka[1]

                if(abs(y_prave_za_x_tacke_linija - y_prave_kretanja_broja) < 2):
                    return True

        return False

'''

    KLASA KOJA PREDSTAVLJA BROJ

'''

from scipy.spatial import distance

class Broj :

    vrednost = -1
    kordinate_sredisnje_tacke= [-1,-1]
    kordinate_prve_tacke = [-1,-1]
    kordinate_druge_tacke = [-1,-1]

    def __init__(self, vrednost, kordinate_sredisnje_tacke):

        self.vrednost = vrednost
        self.kordinate_sredisnje_tacke = kordinate_sredisnje_tacke


    def da_li_je_isti_broj(self, broj):

        if(self.vrednost == broj.vrednost):
            distanca =distance.euclidean(self.kordinate_sredisnje_tacke, broj.kordinate_sredisnje_tacke)
            if(distanca < 60 and distanca > 10):
                return True
            else:
                return False


    def da_li_je_u_izracunatim_brojevima (self, broj):

        y_prave_kretanja_broja_1 = ((self.kordinate_druge_tacke[1] - self.kordinate_prve_tacke[1]) / (
                self.kordinate_druge_tacke[0] - self.kordinate_prve_tacke[0])) * (100 - self.kordinate_prve_tacke[0]) + self.kordinate_prve_tacke[1]

        y_prave_kretanja_broja_2 = ((broj.kordinate_druge_tacke[1] - broj.kordinate_prve_tacke[1]) / (
                broj.kordinate_druge_tacke[0] - broj.kordinate_prve_tacke[0])) * (100 - broj.kordinate_prve_tacke[0]) + broj.kordinate_prve_tacke[1]

        if (abs(y_prave_kretanja_broja_1 - y_prave_kretanja_broja_2) < 5):
            return True
        else:
            return False