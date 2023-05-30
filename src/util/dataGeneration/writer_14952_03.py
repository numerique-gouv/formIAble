"""
Ce writer est dédié au Cerfa N°14952*03 :
VOTE PAR PROCURATION
"""
import random

from .baseWriter import Writer

class Writer14952_03(Writer):
    """
    Ce writer est dédié au Cerfa N°14952*03 :
    VOTE PAR PROCURATION
    """

    def __init__(self, **kwargs):
        """
        :param str num_cerfa: Cerfa number
        """
        super().__init__(**kwargs)
        self.D = {
            "whole_name": self.fake['fr-FR'].name,
            "surname_fields": self.fake['fr-FR'].first_name,
            "name_fields": self.fake['fr-FR'].last_name,
            "date_fields": self.fill_date,
            "numero_electeur": self.fill_numero_electeur,
            "phone": self.fill_phone_number,
            "email": self.fake['fr-FR'].ascii_free_email,
            "city_fields": self.fake['fr-FR'].city,
            "departement": self.fill_departement,
            "pays": self.fake['fr-FR'].country,
            "type_election": self.fill_type_election
        }

    def fill_type_election(self):
        """
        Génère un nom d'élection suivant le format (majuscule ou minuscule):
        ([e-é]lection[s]?)? [pr{}sidentielle"|"l{}gislative"|"europ{}enne"|"r{}gionale"|"municipale"] {s}?

        Note: Les erreurs d'accords sont préservées volontairement
        :return:
        """
        random_e = lambda :  random.choice(["é","e"])
        random_plural = lambda : random.choice(["s",""])

        candidates_1 = "".join([random_e(), "lection",random_plural() ])
        word_1 = random.choice([candidates_1+" ", ""])
        candidates_2 = ["pr{}sidentielle{}","l{}gislative{}", "europ{}enne{}", "r{}gionale{}","municipale{}"]
        word_2_raw = random.choice(candidates_2)
        word_2 = word_2_raw.format(random_plural()) if word_2_raw.startswith("municip") \
                                                    else word_2_raw.format(random_e(), random_plural())

        # On a une chance sur deux d'écrire en majuscules
        if random.getrandbits(1):
            return word_1 + word_2
        else:
            return (word_1 + word_2).upper()

    def fill_numero_electeur(self, **kwargs):
        return self.fake.random_number(digits=9, fix_len=True)