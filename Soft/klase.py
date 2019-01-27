'''

    KLASA KOJA PREDSTAVLJA LINIJU, U OVOM SLUCAJU TO CE BITI PLAVA I ZELENA LINIJA

'''

class Linija:

    __prva_tacka = [0, 0]
    __druga_tacka = [0, 0]

    def __init__(self, kordinate_linije):

        self.__prva_tacka = (kordinate_linije[0], kordinate_linije[1])
        self.__druga_tacka = (kordinate_linije[2], kordinate_linije[3])
