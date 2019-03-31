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
            if(x > druga_tacka[0]):
                y_prave_za_x_tacke_linija = ((self.druga_tacka[1] - self.prva_tacka[1]) / (
                            self.druga_tacka[0] - self.prva_tacka[0])) * (x - self.prva_tacka[0]) + self.prva_tacka[1]

                y_prave_kretanja_broja = ((druga_tacka[1] - prva_tacka[1]) / (
                            druga_tacka[0] - prva_tacka[0])) * (x - prva_tacka[0]) + prva_tacka[1]

                if(abs(y_prave_za_x_tacke_linija - y_prave_kretanja_broja) < 2):
                    return True

        return False

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

    vrednost = -1
    kordinate_sredisnje_tacke= [-1,-1]
    kordinate_prve_tacke = [-1,-1]
    kordinate_druge_tacke = [-1,-1]
    preklopljeni_brojevi = []
    pomeraj_po_x_osi = 0
    pomeraj_po_y_osi = 0

    def __init__(self, vrednost, kordinate_sredisnje_tacke, preslo_sabiranje, preslo_oduzimanje, preklopljeni_brojevi):

        self.vrednost = vrednost
        self.kordinate_sredisnje_tacke = kordinate_sredisnje_tacke
        self.da_li_je_preslo_sabiranje = preslo_sabiranje
        self.da_li_je_preslo_oduzimanje = preslo_oduzimanje
        self.preklopljeni_brojevi = preklopljeni_brojevi
        self.pomeraj_po_x_osi=0
        self.pomeraj_po_y_osi=0


    def da_li_je_isti_broj(self, broj):

        if(self.vrednost == broj.vrednost):
            distanca =distance.euclidean(self.kordinate_sredisnje_tacke, broj.kordinate_sredisnje_tacke)
            if(distanca < 60 and distanca > 10):
                return True
            else:
                return False
