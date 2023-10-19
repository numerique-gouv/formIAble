r"""Module d'accès à un conteneur de données d'un point de connexion S3 appelé s3 pour :
- en lister les fichiers,
- en importer / y exporter les données de modèles LLM (tokens, tokenizers et cache, mais aussi images d'entraînement et de test).

Functions
---------
- getBucketFilesThroughProfile : retourne la liste des fichiers présents dans le conteneur de données.
- exportModelsDataToBucketThroughProfile : exporte l'ensemble des données de modèles présents sur le service Kubernetes exécutant
  ce module dans un conteneur de données en utilisant les informations de connexion d'un profil S3.
- importModelsDataFromBucketThroughProfile : importe l'ensemble des données de modèles du service Kubernetes hébergeant ce module,
  depuis un conteneur de données en utilisant les informations de connexion d'un profil S3.
"""

# importe le module OS pour l'accès aux variables d'environnement
import os
# import le module d'analyse syntaxique de fichiers de configuration
import configparser

# importe le module Boto3 permettant de créer, configurer et gérer des Amazon Web Services (AWS), tels que S3 (Simple Storage Service)
# ou EKS (Elastic Kubernetes Service)
import boto3
# importe le module S3FS permettant de gérer des conteneurs de données (buckets) Amazon S3 (Simple Storage Service)
import s3fs

# importe le module des paramètres et fonctions systèmes
import sys
# importe le module de mesure du temps courant
import time



def getBucketFilesThroughProfile(pBucketNameStr: str = "projet-formiable", pProfileNameStr: str = "projet-formiable"):
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
    # si le fichier des informations de connexion aux services AWS fait partie de la liste (renvoyée par read) des fichiers analysés avec succès
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
        # dictionnaire des paramètres de connexion au point de connexion S3 du Datalab SSP cloud :
        # celui-ci est obtenu par fusion du dictionnaire contenant l'URL du point de connexion S3 avec
        # le dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudS3FileSystemConnectionParameters = {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"} | lDatalabSSPcloudAWScredentialsDictionary
        # système de fichier du point de connexion S3
        lDatalabSSPcloudS3FileSystem = s3fs.S3FileSystem(client_kwargs=lDatalabSSPcloudS3FileSystemConnectionParameters)
#        lDatalabSSPcloudS3FileSystem.rm(f"s3://{pBucketNameStr}/data/models/fakeModel/", recursive=True)
        # retourne la liste les fichiers du conteneur de données pBucketNameStr
        return lDatalabSSPcloudS3FileSystem.find(f"s3://{pBucketNameStr}/")
    else:
        return ["Erreur lors de l'analyse du fichier des informations de connexion aux services AWS"]


def exportModelsDataToBucketThroughProfile(
    pModelsLocalCacheRootDirectoryStr: str,
    pModelsLocalDataRootDirectoryStr: str,
    pModelsNamesStrs,
    pBucketNameStr: str = "projet-formiable",
    pProfileNameStr: str = "projet-formiable"
):
    r"""Exporte l'ensemble des données (tokens, tokenizers et cache, mais aussi images d'entraînement et de test) des modèles `pModelsNamesStrs`
    présents dans les répertoires `pModelsLocalCacheRootDirectoryStr` et `pModelsLocalDataRootDirectoryStr` sur le service Kubernetes exécutant
    ce module, dans le conteneur de données `pBucketNameStr` du point de connexion S3 en utilisant les informations de connexion du profil
    `pProfileNameStr`.

    Parameters
    ----------
    pModelsLocalCacheRootDirectoryStr : str
        le répertoire local racine du cache des modèles à exporter.
    pModelsLocalDataRootDirectoryStr : str
        le répertoire local racine des données des modèles à exporter.
    pModelsNamesStrs : list of str
        la liste des noms des modèles à exporter.
    pBucketNameStr : str, default="projet-formiable"
        le nom du conteneur de données du point de connexion S3 dans lequel exporter les fichiers.
    pProfileNameStr : str, default="projet-formiable"
        le nom du profil S3 dont les informations de connexion sont utilisées pour accéder au conteneur de données `pBucketNameStr`.
    """
    lConfigurationParser = configparser.ConfigParser()
    # si le fichier des informations de connexion aux services AWS fait partie de la liste (renvoyée par read) des fichiers analysés avec succès
    # par l'analyseur syntaxique
    if os.environ['AWS_SHARED_CREDENTIALS_FILE'] in lConfigurationParser.read(os.environ['AWS_SHARED_CREDENTIALS_FILE']):
        # dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudAWScredentialsDictionary = dict(lConfigurationParser.items(pProfileNameStr))
        # dictionnaire des paramètres de connexion au point de connexion S3 du Datalab SSP cloud :
        # celui-ci est obtenu par fusion du dictionnaire contenant l'URL du point de connexion S3 avec
        # le dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudS3FileSystemConnectionParameters = {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"} | lDatalabSSPcloudAWScredentialsDictionary
        # système de fichier du point de connexion S3
        lDatalabSSPcloudS3FileSystem = s3fs.S3FileSystem(client_kwargs=lDatalabSSPcloudS3FileSystemConnectionParameters)

        # crée le répertoire racine des modèles si non existant dans le conteneur de données pBucketNameStr du point de connexion S3
        lDatalabSSPcloudS3ModelsRootDirectoryStr = f"s3://{pBucketNameStr}/data/models"
        if not lDatalabSSPcloudS3FileSystem.exists(lDatalabSSPcloudS3ModelsRootDirectoryStr):
            lDatalabSSPcloudS3FileSystem.mkdirs(lDatalabSSPcloudS3ModelsRootDirectoryStr)

        # copie les tokens, tokenizers et cache de chaque modèle dans le répertoire racine des modèles sur le conteneur de données pBucketNameStr
        # du point de connexion S3
        for lModelNameStr in pModelsNamesStrs:
            # si le répertoire du modèle n'existe pas déjà dans le conteneur de données pBucketNameStr du point de connexion S3,
            if not lDatalabSSPcloudS3FileSystem.exists(f"{lDatalabSSPcloudS3ModelsRootDirectoryStr}/{lModelNameStr}"):
                print(f"Export des tokens, tokenizers et cache du modèle {lModelNameStr}")
                # en stocke les tokens, tokenizers et cache sur le point de connexion S3
                # durée de la copie = environ 38 secondes pour 2.4 Go
                lDatalabSSPcloudS3FileSystem.put(
                    f"{pModelsLocalCacheRootDirectoryStr}/{lModelNameStr}",
                    f"{lDatalabSSPcloudS3ModelsRootDirectoryStr}/",
                    recursive=True
                )

        # si le répertoire racine des données / fichiers d'entraînement et de test des modèles n'existe pas déjà dans le conteneur de
        # données pBucketNameStr du point de connexion S3,
        lDatalabSSPcloudS3ModelsDataRootDirectoryStr = f"s3://{pBucketNameStr}/data/ls_data/divers"
        if not lDatalabSSPcloudS3FileSystem.exists(lDatalabSSPcloudS3ModelsDataRootDirectoryStr):
            # le crée
            lDatalabSSPcloudS3FileSystem.mkdirs(lDatalabSSPcloudS3ModelsDataRootDirectoryStr)

        # itère sur les entrées (répertoires et fichiers) du répertoire racine des données des modèles à copier
        # durée de la copie = environ 40 secondes
        with os.scandir(pModelsLocalDataRootDirectoryStr) as lModelsDataToBeCopiedRootDirectoryIterator:
            print("Export des fichiers d'entraînement et de test des modèles")
            # pour chaque entrée,
            for lModelDataToBeCopiedItem in lModelsDataToBeCopiedRootDirectoryIterator:
                # si ladite entrée n'existe pas déjà dans le répertoire racine du cache des modèles sur le conteneur de données pBucketNameStr du
                # point de connexion S3
                if not lDatalabSSPcloudS3FileSystem.exists(f"{lDatalabSSPcloudS3ModelsDataRootDirectoryStr}/{lModelDataToBeCopiedItem.name}"):
                    # et si ladite entrée est un répertoire qui n'est pas un lien symbolique (c'est-à-dire un répertoire contenant effectivement
                    # le cache d'un modèle)
                    if lModelDataToBeCopiedItem.is_dir(follow_symlinks=False):
                        # copie les données des modèles stockées dans ledit répertoire vers le répertoire racine des données des modèles
                        # sur le conteneur de données pBucketNameStr du point de connexion S3
                        lDatalabSSPcloudS3FileSystem.put(
                            f"{lModelDataToBeCopiedItem.path}",
                            f"{lDatalabSSPcloudS3ModelsDataRootDirectoryStr}/",
                            recursive=True
                        )


def importModelsDataFromBucketThroughProfile(
    pModelsLocalCacheRootDirectoryStr: str,
    pModelsLocalDataRootDirectoryStr: str,
    pModelsNamesStrs,
    pBucketNameStr: str = "projet-formiable",
    pProfileNameStr: str = "projet-formiable"
):
    r"""Importe l'ensemble des données (tokens, tokenizers et cache, mais aussi images d'entraînement et de test) des modèles `pModelsNamesStrs`
    dans les répertoires `pModelsLocalDataRootDirectoryStr` et `pModelsLocalCacheRootDirectoryStr` du service Kubernetes hébergeant ce module,
    depuis le conteneur de données `pBucketNameStr` du point de connexion S3 en utilisant les informations de connexion du profil `pProfileNameStr`.

    Parameters
    ----------
    pModelsLocalCacheRootDirectoryStr : str
        le répertoire local racine dans lequel importer le cache des modèles.
    pModelsLocalDataRootDirectoryStr : str
        le répertoire local racine dans lequel importer les données des modèles.
    pModelsNamesStrs : list of str
        la liste des noms des modèles à importer.
    pBucketNameStr : str, default="projet-formiable"
        le nom du conteneur de données du point de connexion S3 depuis lequel importer les fichiers.
    pProfileNameStr : str, default="projet-formiable"
        le nom du profil S3 dont les informations de connexion sont utilisées pour accéder au conteneur de données `pBucketNameStr`.
    """
    lConfigurationParser = configparser.ConfigParser()
    # si le fichier des informations de connexion aux services AWS fait partie de la liste (renvoyée par read) des fichiers analysés avec succès
    # par l'analyseur syntaxique
    if os.environ['AWS_SHARED_CREDENTIALS_FILE'] in lConfigurationParser.read(os.environ['AWS_SHARED_CREDENTIALS_FILE']):
        # dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudAWScredentialsDictionary = dict(lConfigurationParser.items(pProfileNameStr))
        # dictionnaire des paramètres de connexion au point de connexion S3 du Datalab SSP cloud :
        # celui-ci est obtenu par fusion du dictionnaire contenant l'URL du point de connexion S3 avec
        # le dictionnaire des informations de connexion définies pour le profil pProfileNameStr
        lDatalabSSPcloudS3FileSystemConnectionParameters = {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"} | lDatalabSSPcloudAWScredentialsDictionary
        # système de fichier du point de connexion S3
        lDatalabSSPcloudS3FileSystem = s3fs.S3FileSystem(client_kwargs=lDatalabSSPcloudS3FileSystemConnectionParameters)

        # copie les fichiers générés et de cache de chaque modèle depuis le répertoire du modèle présent dans le conteneur de données pBucketNameStr
        # du point de connexion S3
        for lModelNameStr in pModelsNamesStrs:
            # si le répertoire du modèle existe sur le conteneur de données pBucketNameStr du point de connexion S3,
            lDatalabSSPcloudS3ModelDirectoryStr = f"s3://{pBucketNameStr}/data/models/{lModelNameStr}"
            if lDatalabSSPcloudS3FileSystem.exists(lDatalabSSPcloudS3ModelDirectoryStr):
                print(f"Import des tokens, tokenizers et cache du modèle {lModelNameStr}")
                # en copie les tokens, tokenizers et cache dans le répertoire pModelsLocalCacheRootDirectoryStr
                # durée de la copie = environ 43 secondes pour 2.4 Go
                lDatalabSSPcloudS3FileSystem.get(
                    lDatalabSSPcloudS3ModelDirectoryStr,
                    f"{pModelsLocalCacheRootDirectoryStr}/",
                    recursive=True
                )

        # si le chemin racine des fichiers d'entraînement et de test des modèles existe sur le conteneur de données pBucketNameStr du point de
        # connexion S3 (ledit chemin doit terminer par "/" sinon la copie via get rajoute le dernier répertoire mentionné dans le chemin),
        lDatalabSSPcloudS3ModelsDataRootDirectoryStr = f"s3://{pBucketNameStr}/data/ls_data/divers/"
        if lDatalabSSPcloudS3FileSystem.exists(lDatalabSSPcloudS3ModelsDataRootDirectoryStr):
            print("Import des fichiers d'entraînement et de test des modèles")
            # copie les fichiers d'entraînement et de test des modèles depuis leur répertoire racine présent sur le conteneur de données
            # pBucketNameStr du point de connexion S3 vers le répertoire pModelsLocalDataRootDirectoryStr
            # durée de la copie = environ 37 secondes
            lDatalabSSPcloudS3FileSystem.get(
                lDatalabSSPcloudS3ModelsDataRootDirectoryStr,
                f"{pModelsLocalDataRootDirectoryStr}/",
                recursive=True
            )



if __name__ == "__main__":
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

    print("Version des modules", "\n- Boto3 :", boto3.__version__, "\n- S3FS :", s3fs.__version__, "\n")
    print("Chemin complet de l'exécutable python =", sys.executable)
    print("Chemins système =", sys.path, "\n")
    print("AWS :\n- Credentials file =", os.environ["AWS_SHARED_CREDENTIALS_FILE"], "\n- S3 end point =", os.environ["AWS_S3_ENDPOINT"], "\n")

    print("Liste des fichiers du conteneur de données personnel")
    # session S3 utilisant les informations de connexion indiquées dans le profil personnel du fichier desdites informations de connexion aux services S3
    # !!! Attention !!! Une session est un objet non thread-safe : il convient donc de créer une session par thread dans le cadre d'une exécution
    # multi-thread
    datalabSSPcloudAWSSession = boto3.Session(profile_name="personal")
    # ressource (non thread-safe) S3 haut niveau liée au point de connexion S3 du Datalab SSP cloud
    # !!! Attention !!! Une ressource est un objet non thread-safe : il convient donc de créer une ressource par thread dans le cadre d'une exécution
    # multi-thread
    datalabSSPcloudAWSs3Resource = datalabSSPcloudAWSSession.resource(
        service_name="s3",
        endpoint_url=f"https://{os.environ['AWS_S3_ENDPOINT']}"
    )
    # liste les 3 premiers objets contenus dans chaque conteneur de données S3 récupéré sur le point de connexion
    for lS3BucketIdx, lS3Bucket in enumerate(datalabSSPcloudAWSs3Resource.buckets.all(), start=1):
        #print(f"- Bucket S3 n° {lS3BucketIdx} : {lS3Bucket.name}")
        print(f"- Conteneur de données S3 n° {lS3BucketIdx} : {lS3Bucket.name}")
        for lS3BucketObjectIdx, lS3BucketObject in enumerate(lS3Bucket.objects.limit(3), start=1):
            #print(f"  * Object n° {lS3BucketObjectIdx} : {lS3BucketObject.key}")
            print(f"  * Objet n° {lS3BucketObjectIdx} : {lS3BucketObject.key}")

    print("\nListe des fichiers du conteneur de données \"projet-formIAble\"")
    print("-", "\n- ".join(getBucketFilesThroughProfile(pBucketNameStr="projet-formiable", pProfileNameStr="projet-formiable")), "\n")

    print("Import des modèles et des fichiers d'entraînement et de test depuis le conteneur de données partagé \"projet-formIAble\" du point de connexion S3 vers le système de fichier")
    lBeforeCopyTime = time.time()
    importModelsDataFromBucketThroughProfile(
        pModelsLocalCacheRootDirectoryStr = "./data/models",
        pModelsLocalDataRootDirectoryStr = "./src/data/synthetic_forms",
        pModelsNamesStrs = ["donut_trained"],
        pBucketNameStr = "projet-formiable",
        pProfileNameStr = "projet-formiable"
    )
    print("==> Durée de la copie des modèles et des fichiers d'entraînement et de test =", time.time() - lBeforeCopyTime, "secondes\n")

#    print("Export des modèles et des fichiers d'entraînement et de test depuis le système de fichier vers le conteneur de données personnel du point de connexion S3")
#    lBeforeCopyTime = time.time()
#    exportModelsDataToBucketThroughProfile(
#        pModelsLocalCacheRootDirectoryStr = "./data/models",
#        pModelsLocalDataRootDirectoryStr = "./src/data/synthetic_forms",
#        pModelsNamesStrs = ["donut_trained"],
#        pBucketNameStr = "myPersonalBucket",
#        pProfileNameStr = "personal"
#    )
#    print("==> Durée de la copie des modèles et des fichiers d'entraînement et de test =", time.time() - lBeforeCopyTime, "secondes")
