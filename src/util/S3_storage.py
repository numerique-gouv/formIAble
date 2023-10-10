# importe le module OS pour l'accès aux variables d'environnement
import os
# import le module d'analyse syntaxique de fichiers de configuration
import configparser

# importe le module Boto3 permettant de créer, configurer et gérer des Amazon Web Services (AWS), tels que S3 (Simple Storage Service) ou EKS (Elastic Kubernetes Service)
import boto3
# importe le module S3FS permettant de gérer des conteneurs de données (buckets) Amazon S3 (Simple Storage Service)
import s3fs

# importe le module des paramètres et fonctions systèmes
import sys
# importe le module de mesure du temps courant
import time



# définit les variables d'environnement permettant la connexion à l'espace de stockage S3
# chemin complet du fichier des informations de connexion aux services AWS, contenant au moins un profil (default) et, pour chaque profil,
# les valeurs des informations de connexion au stockage du compte Datalab SSP cloud (que l'on retrouve dont l'onglet éponyme de "Mon compte") :
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_SESSION_TOKEN
# Les valeurs du profil sélectionné sont utilisées lors de la connexion aux services web Amazon (AWS)
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "./security/AWScredentials.info"
# point de connexion S3 du Datalab SSP cloud, via la suite de stockage d'objets minIO, basée sur le cloud et compatible avec l'API S3 d'Amazon
# déjà transmis par Onyxia
#os.environ["AWS_S3_ENDPOINT"] = "minio.lab.sspcloud.fr"



def getBucketFilesThroughProfile(pBucketNameStr : str = "projet-formiable", pProfileNameStr : str = "projet-formiable"):
    r"""Retourne la liste des fichiers présents dans le conteneur de données `pBucketNameStr` du point de connexion S3 en utilisant les
    informations de connexion du profil `pProfileNameStr`.

    Parameters
    ----------
    pBucketNameStr : str, default="projet-formiable"
        le nom du conteneur de données du point de connexion S3 dont les fichiers doivent être listés.
    pProfileNameStr : str, default="projet-formiable"
        le nom du profil S3 dont les informations de connexion sont utilisées pour accéder au conteneur de données `pBucketNameStr`.

    Returns
    -------
    list of S3 files/strings
        la liste des fichiers présents dans le conteur de données, ou un message d'erreur encapsulé dans une liste si le profil `pProfileNameStr`
        ne figure pas parmi les profils présents dans le fichier ``AWScredentials.info`` des profils.
    """
    lConfigurationParser = configparser.ConfigParser()
    # si le fichier des informations de connexion aux services AWS fait partie de la liste des fichiers analysés avec succès (renvoyée par read)
    # par l'analyseur syntaxique
    if os.environ['AWS_SHARED_CREDENTIALS_FILE'] in lConfigurationParser.read(os.environ['AWS_SHARED_CREDENTIALS_FILE']):
        # dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudAWScredentialsDictionary = dict(lConfigurationParser.items(pProfileNameStr))
#        for lDatalabSSPcloudAWScredentialsDictionaryKey, lDatalabSSPcloudAWScredentialsDictionaryValue in lDatalabSSPcloudAWScredentialsDictionary.items():
#            # !!! l'instanciation du système de fichiers S3 via s3fs.S3FileSystem() ne reconnaît pas la variable d'environnement AWS_ACCESS_KEY_ID...
#            lEnvironmentVariableNameStr = lDatalabSSPcloudAWScredentialsDictionaryKey.upper()
#            os.environ[lEnvironmentVariableNameStr] = lDatalabSSPcloudAWScredentialsDictionaryValue
#            print("OS variable", lEnvironmentVariableNameStr, "=", lDatalabSSPcloudAWScredentialsDictionaryValue)
#            print("OS variable", lEnvironmentVariableNameStr, "=", os.environ[lEnvironmentVariableNameStr])
        # dictionnaire des paramètres de connexion au point de connexion S3 du Datalab SSP cloud
        # celui-ci est obtenu par fusion du dictionnaire contenant l'URL du point de connexion S3 avec
        # le dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudS3FileSystemConnectionParameters = {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"} | lDatalabSSPcloudAWScredentialsDictionary
        # système de fichier du point de connexion S3
        lDatalabSSPcloudS3FileSystem = s3fs.S3FileSystem(client_kwargs=lDatalabSSPcloudS3FileSystemConnectionParameters)
#        lDatalabSSPcloudS3FileSystem.rm(f"s3://{pBucketNameStr}/LLM/fakeModel/", recursive=True)
        # retourne la liste les fichiers du conteneur de données pBucketNameStr
        return lDatalabSSPcloudS3FileSystem.find(f"s3://{pBucketNameStr}/")
    else:
        return "Erreur lors de l'analyse du fichier des informations de connexion aux services AWS"


def exportModelsDataToBucketThroughProfile(pModelsNamesStrs, pBucketNameStr : str = "projet-formiable", pProfileNameStr : str = "projet-formiable"):
    r"""Exporte l'ensemble des données (tokens, tokenizers et cache, mais aussi images d'entraînement et de test) des modèles `pModelsNamesStrs`
    présents sur le service Kubernetes exécutant ce notebook dans le conteneur de données `pBucketNameStr` du point de connexion S3 en utilisant les
    informations de connexion du profil `pProfileNameStr`.

    Parameters
    ----------
    pModelsNamesStrs : list of str
        la liste des noms des modèles à exporter
    pBucketNameStr : str, default="projet-formiable"
        le nom du conteneur de données du point de connexion S3 sur lequel les fichiers doivent être exportés.
    pProfileNameStr : str, default="projet-formiable"
        le nom du profil S3 dont les informations de connexion sont utilisées pour accéder au conteneur de données `pBucketNameStr`.
    """
    lConfigurationParser = configparser.ConfigParser()
    # si le fichier des informations de connexion aux services AWS fait partie de la liste des fichiers analysés avec succès (renvoyée par read)
    # par l'analyseur syntaxique
    if os.environ['AWS_SHARED_CREDENTIALS_FILE'] in lConfigurationParser.read(os.environ['AWS_SHARED_CREDENTIALS_FILE']):
        # dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudAWScredentialsDictionary = dict(lConfigurationParser.items(pProfileNameStr))
        # dictionnaire des paramètres de connexion au point de connexion S3 du Datalab SSP cloud
        # celui-ci est obtenu par fusion du dictionnaire contenant l'URL du point de connexion S3 avec
        # le dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudS3FileSystemConnectionParameters = {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"} | lDatalabSSPcloudAWScredentialsDictionary
        # système de fichier du point de connexion S3
        lDatalabSSPcloudS3FileSystem = s3fs.S3FileSystem(client_kwargs=lDatalabSSPcloudS3FileSystemConnectionParameters)

        # crée le répertoire data/models si non existant dans le conteneur de données pBucketNameStr du point de connexion S3
        if not lDatalabSSPcloudS3FileSystem.exists(f"s3://{pBucketNameStr}/data/models/"):
            lDatalabSSPcloudS3FileSystem.mkdirs(f"s3://{pBucketNameStr}/data/models/")

        # copie les données de chaque modèle dans le répertoire LLM du conteneur de données pBucketNameStr du point de connexion S3
        for lLlmModelNameStr in pModelsNamesStrs:
            # si le répertoire du modèle n'existe pas déjà,
            if not lDatalabSSPcloudS3FileSystem.exists(f"s3://{pBucketNameStr}/data/models/{lLlmModelNameStr}/"):
                print(f"Export des données du modèle {lLlmModelNameStr}")
                # en stocke toutes les données sur le point de connexion S3
                # durée de la copie = environ 38 secondes pour 2.4 Go
                lDatalabSSPcloudS3FileSystem.put(
                    f"./src/models/{lLlmModelNameStr}",
                    f"s3://{pBucketNameStr}/data/models/",
                    recursive=True
                )


def importModelsDataToBucketThroughProfile(pModelsNamesStrs, pBucketNameStr : str = "projet-formiable", pProfileNameStr : str = "projet-formiable"):
    r"""Importe l'ensemble des données (tokens, tokenizers et cache, mais aussi images d'entraînement et de test) des modèles `pModelsNamesStrs`
    dans des sous-répertoires de ``./src`` du service Kubernetes exécutant ce notebook, depuis le conteneur de données `pBucketNameStr` du point
    de connexion S3 en utilisant les informations de connexion du profil `pProfileNameStr`.

    Parameters
    ----------
    pModelsNamesStrs : list of str
        la liste des noms des modèles à importer
    pBucketNameStr : str, default="projet-formiable"
        le nom du conteneur de données du point de connexion S3 depuis lequel les fichiers doivent être importés.
    pProfileNameStr : str, default="projet-formiable"
        le nom du profil S3 dont les informations de connexion sont utilisées pour accéder au conteneur de données `pBucketNameStr`.
    """
    lConfigurationParser = configparser.ConfigParser()
    # si le fichier des informations de connexion aux services AWS fait partie de la liste des fichiers analysés avec succès (renvoyée par read)
    # par l'analyseur syntaxique
    if os.environ['AWS_SHARED_CREDENTIALS_FILE'] in lConfigurationParser.read(os.environ['AWS_SHARED_CREDENTIALS_FILE']):
        # dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudAWScredentialsDictionary = dict(lConfigurationParser.items(pProfileNameStr))
        # dictionnaire des paramètres de connexion au point de connexion S3 du Datalab SSP cloud
        # celui-ci est obtenu par fusion du dictionnaire contenant l'URL du point de connexion S3 avec
        # le dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudS3FileSystemConnectionParameters = {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"} | lDatalabSSPcloudAWScredentialsDictionary
        # système de fichier du point de connexion S3
        lDatalabSSPcloudS3FileSystem = s3fs.S3FileSystem(client_kwargs=lDatalabSSPcloudS3FileSystemConnectionParameters)

        # copie les fichiers générés de chaque modèle depuis le répertoire du modèle présent dans le conteneur de données pBucketNameStr du point de connexion S3
        for lLlmModelNameStr in pModelsNamesStrs:
            # si le répertoire du modèle existe sur le conteneur de données pBucketNameStr du point de connexion S3,
            if lDatalabSSPcloudS3FileSystem.exists(f"s3://{pBucketNameStr}/data/models/{lLlmModelNameStr}/"):
                print(f"Import des données du modèle {lLlmModelNameStr}")
                # en copie les données dans le répertoire ../models
                # durée de la copie = environ 43 secondes pour 2.4 Go
                lDatalabSSPcloudS3FileSystem.get(
                    f"s3://{pBucketNameStr}/data/models/{lLlmModelNameStr}",
                    "./src/models/",
                    recursive=True
                )

        # si le répertoire racine des fichiers d'entraînement et de test des modèles existe sur le conteneur de données pBucketNameStr du point de connexion S3,
        if lDatalabSSPcloudS3FileSystem.exists(f"s3://{pBucketNameStr}/data/ls_data/divers"):
            print("Import des fichiers d'entraînement et de test des modèles")
            # copie les fichiers d'entraînement et de test des modèles depuis le répertoire data/ls_data/divers du conteneur de données pBucketNameStr
            # du point de connexion S3 dans le répertoire ./src/data/synthetic_forms
            # durée de la copie = environ 37 secondes
            lDatalabSSPcloudS3FileSystem.get(
                f"s3://{pBucketNameStr}/data/ls_data/divers",
                "./src/data/synthetic_forms/",
                recursive=True
            )



if __name__ == "__main__":
    print("Version des modules", "\n- Boto3 :", boto3.__version__, "\n- S3FS :", s3fs.__version__, "\n")
    print("Exécutable python =", sys.executable)
    print("Chemins système =", sys.path, "\n")
    print("AWS :\n- Credentials file =", os.environ["AWS_SHARED_CREDENTIALS_FILE"], "\n- S3 end point =", os.environ["AWS_S3_ENDPOINT"], "\n")
    print("Liste des fichiers du conteneur de données")
    print("-", "\n- ".join(getBucketFilesThroughProfile(pBucketNameStr="projet-formiable", pProfileNameStr="projet-formiable")), "\n")
    print("Import des modèles et des images d'entraînement et de test depuis le conteneur de données partagé \"projet-formIAble\" du point de connexion S3 vers le système de fichier")
    lBeforeCopyTime = time.time()
    importModelsDataToBucketThroughProfile(
        pModelsNamesStrs = ["donut_trained"],
        pBucketNameStr = "projet-formiable",
        pProfileNameStr = "projet-formiable"
    )
    print("==> Durée de la copie des modèles et des images d'entraînement et de test =", time.time() - lBeforeCopyTime, "secondes\n")
#    print("Export des modèles depuis le système de fichier vers le conteneur de données personnel du point de connexion S3")
#    lBeforeCopyTime = time.time()
#    exportModelsDataToBucketThroughProfile(
#        pModelsNamesStrs = ["donut_trained"],
#        pBucketNameStr = "myPersonalBucket",
#        pProfileNameStr = "personal"
#    )
#    print("==> Durée de la copie des modèles =", time.time() - lBeforeCopyTime, "secondes")
