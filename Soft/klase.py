'''

    KLASA KOJA PREDSTAVLJA LINIJU, U OVOM SLUCAJU TO CE BITI PLAVA I ZELENA LINIJA

'''

class Linija:

   # __prva_tacka = [0, 0]
  #  __druga_tacka = [0, 0]

    def __init__(self, kordinate_linije):

        self.prva_tacka = (kordinate_linije[0], kordinate_linije[1])
        self.druga_tacka = (kordinate_linije[2], kordinate_linije[3])






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
    da_li_je_preslo_sabiranje = False
    da_li_je_preslo_oduzimanje = False
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
            if(distanca < 100):
                return True
            else:
                return False
