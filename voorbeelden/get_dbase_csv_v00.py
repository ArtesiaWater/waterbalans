import platform
assert platform.architecture()[0] == "32bit", "Error: script requires 32bit Python!"

import pyodbc
import pandas as pd
import os

outdir = (r"C:\Users\dbrak\Documents\01-Projects\17026004_WATERNET_Waterbalansen"
          r"\03data\DataExport_frompython")

# %% Set up database connection
connStr = (
    "DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    "DBQ=C:\\Users\dbrak\Documents\\01-Projects\\17026004_WATERNET_Waterbalansen"
    "\\03data\\20181106_waterbalans.accdb;"
    )
cnxn = pyodbc.connect(connStr)
cur = cnxn.cursor()

# %% Get EAG codes
EAG = cur.execute("SELECT * FROM EAG")
EAG_list = EAG.fetchall()

# %% Get opp files
for iEAG in EAG_list:
    GETOPPSQL = ("select EAG.ID as EAGID, " 
                "EAG.Code as EAGCode, "
                "Bakjes.ID as BakjeID, "
                "Bakjes.Omschrijving as BakjeOmschrijving, "
                "BakjeTypen.PyCode as BakjePyCode, "
                "BakjeParamWaarden.Waarde as OppWaarde "
                "from (((EAG inner join Bakjes on Bakjes.EAGID = EAG.ID) "
                "inner join BakjeParamWaarden on BakjeParamWaarden.BakjeID = Bakjes.ID) "
                "inner join BakjeParams on BakjeParams.ID = BakjeParamWaarden.BakjeParamID) "
                "inner join BakjeTypen on BakjeTypen.ID = Bakjes.BakjeTypeID "
                "where BakjeParams.Code = 'Opp' and EAG.ID = {}".format(iEAG[0]))

    df = pd.read_sql(GETOPPSQL, cnxn)
    if not df.empty:
        df.to_csv(os.path.join(outdir, "opp_{0}_{1}.csv".format(iEAG[0], iEAG[1])), sep=";")

# %% get param files
for iEAG in EAG_list:
    # GETPARAMSQL = ("SELECT EAG.ID as EAGID, "
    #             "EAG.Code as EAGCode, "
    #             "Bakjes.ID AS BakjeID, "
    #             "LaagTypen.Laagvolgorde, "
    #             "Laagparams.Code as ParamCode, "
    #             "Laagparamwaarden.Waarde, "
    #             "-9999 AS BakjeParamWaardenID, "
    #             "LaagParamWaarden.ID AS LaagParamWaardenID FROM "
    #             "(((((((Bakjes INNER JOIN Lagen ON Bakjes.ID = Lagen.BakjeID) "
    #             "INNER JOIN LaagTypen ON Lagen.LaagTypeID = LaagTypen.ID) "
    #             "INNER JOIN laagparamwaarden ON Laagparamwaarden.LaagID = Lagen.ID) "
    #             "INNER JOIN Laagparams ON Laagparams.ID = Laagparamwaarden.Laagparamid) "
    #             "INNER JOIN ReeksTypen ON ReeksTypen.ID = laagreekswaarden.reekstypeid) "
    #             "INNER JOIN EAG ON EAG.id = Bakjes.EAGID) "
    #             "INNER JOIN config ON config.ID = EAG.configID) "
    #             "INNER JOIN GAF ON GAF.id = config.GAFID where EAG.ID = {}".format(iEAG[0]))
    GETPARAMSQL = ("SELECT EAG.ID as EAGID, "
                    "EAG.Code as EAGCode, "
                    "Bakjes.ID AS BakjeID, "
                    "LaagTypen.Laagvolgorde, "
                    "Laagparams.Code as ParamCode, "
                    "Laagparamwaarden.Waarde, "
                    "-9999 AS BakjeParamWaardenID, "
                    "LaagParamWaarden.ID AS LaagParamWaardenID FROM "
                    "((((((Bakjes INNER JOIN Lagen ON Bakjes.ID = Lagen.BakjeID) "
                    "INNER JOIN LaagTypen ON Lagen.LaagTypeID = LaagTypen.ID) "
                    "INNER JOIN laagparamwaarden ON Laagparamwaarden.LaagID = Lagen.ID) "
                    "INNER JOIN Laagparams ON Laagparams.ID = Laagparamwaarden.Laagparamid) "
                    "INNER JOIN EAG ON EAG.id = Bakjes.EAGID) "
                    "INNER JOIN config ON config.ID = EAG.configID) "
                    "INNER JOIN GAF ON GAF.id = config.GAFID where EAG.ID = {}".format(iEAG[0]))

    df0 = pd.read_sql(GETPARAMSQL, cnxn)

    # GETPARAMSQL = ("SELECT EAG.ID as EAGID, "
    #                "EAG.Code as EAGCode, "
    #                "Bakjes.ID AS BakjeID, "
    #                "-9999 as Laagvolgorde, "
    #                "BakjeParams.Code, "
    #                "Bakjeparamwaarden.Waarde, "
    #                "BakjeParamWaarden.ID AS BakjeParamWaardenID, "
    #                "-9999 AS LaagParamWaardenID FROM "
    #                "((((((Bakjes INNER JOIN BakjeTypen ON Bakjes.BakjeTypeID = BakjeTypen.ID) "
    #                "INNER JOIN bakjeparamwaarden ON bakjeparamwaarden.BakjeID = Bakjes.ID) "
    #                "INNER JOIN Bakjeparams ON Bakjeparams.ID = Bakjeparamwaarden.bakjeparamid) "
    #                "INNER JOIN ReeksTypen ON ReeksTypen.ID = bakjeparamwaarden.reekstypeid) "
    #                "INNER JOIN EAG ON EAG.id = Bakjes.EAGID) "
    #                "INNER JOIN config ON config.ID = EAG.configID) "
    #                "INNER JOIN GAF ON GAF.id = config.GAFID where "
    #                "BakjeParams.Code <> 'Opp' and EAG.ID = {}".format(iEAG[0]))

    GETPARAMSQL = ("SELECT EAG.ID as EAGID, "
                   "EAG.Code as EAGCode, "
                   "Bakjes.ID AS BakjeID, "
                   "-9999 as Laagvolgorde, "
                   "BakjeParams.Code AS ParamCode, "
                   "Bakjeparamwaarden.Waarde, "
                   "BakjeParamWaarden.ID AS BakjeParamWaardenID, "
                   "-9999 AS LaagParamWaardenID FROM "
                   "(((((Bakjes INNER JOIN BakjeTypen ON Bakjes.BakjeTypeID = BakjeTypen.ID) "
                   "INNER JOIN bakjeparamwaarden ON bakjeparamwaarden.BakjeID = Bakjes.ID) "
                   "INNER JOIN Bakjeparams ON Bakjeparams.ID = Bakjeparamwaarden.bakjeparamid) "
                   "INNER JOIN EAG ON EAG.id = Bakjes.EAGID) "
                   "INNER JOIN config ON config.ID = EAG.configID) "
                   "INNER JOIN GAF ON GAF.id = config.GAFID where "
                   "BakjeParams.Code <> 'Opp' and EAG.ID = {}".format(iEAG[0]))

    df1 = pd.read_sql(GETPARAMSQL, cnxn)

    # TODO: test is this is what R function 'rbind' does
    df = pd.concat([df0, df1], axis=0)

    if not df.empty:
        df.to_csv(os.path.join(outdir, "param_{0}_{1}.csv".format(iEAG[0], iEAG[1])), sep=";")

# %% get reeksen files
for iEAG in EAG_list:
    GETREEKSEAGSQL = ("SELECT EAG.ID AS EAGID, "
                        "EAG.Code AS EAGCode, "
                        "-9999 AS BakjeID, "
                        "-9999 as Laagvolgorde, "
                        "EAGClusterTypen.Code AS ClusterType, "
                        "EAGReeksen.Code AS ReeksType, "
                        "ReeksTypen.ReeksType as ParamType, "
                        "EAGReeksWaarden.Waarde AS Waarde, "
                        "EAGReeksWaarden.WaardeAlfa AS WaardeAlfa, "
                        "EAGReeksWaarden.StartDag, "
                        "EAGReeksWaarden.ID as EAGReeksWaardeID, "
                        "-9999 as BakjeReeksWaardeID, "
                        "-9999 as LaagReeksWaardeID FROM "
                        "(((((((EAG INNER JOIN Config ON Config.ID = EAG.ConfigID) "
                        "INNER JOIN GAF ON GAF.ID = Config.GAFID) "
                        "INNER JOIN EAGClusters ON EAGClusters.EAGID = EAG.ID) "
                        "INNER JOIN EAGReeksWaarden ON EAGReeksWaarden.EAGClusterID = EAGClusters.ID) "
                        "INNER JOIN EAGClusterTypen on EAGClusters.EAGClusterTypeID = EAGClusterTypen.ID) "
                        "INNER JOIN EAGReeksen ON EAGReeksen.ID = EAGReeksWaarden.EAGReeksID) "
                        "INNER JOIN ReeksTypen ON ReeksTypen.ID = EAGReeksWaarden.ReeksTypeID) "
                        "WHERE EAG.ID = {}".format(iEAG[0]))
    df0 = pd.read_sql(GETREEKSEAGSQL, cnxn)
    
    GETREEKSBAKJESQL = ("SELECT EAG.ID AS EAGID, "
                        "EAG.Code AS EAGCode, "
                        "Bakjes.ID AS BakjeID, "
                        "-9999 as Laagvolgorde, "
                        "BakjeClusterTypen.Code AS ClusterType, "
                        "BakjeReeksen.Code AS ReeksType, "
                        "-9999 AS ParamType, "
                        "BakjeReeksWaarden.Waarde AS Waarde, "
                        "BakjeReeksWaarden.WaardeAlfa AS WaardeAlfa, "
                        "BakjeReeksWaarden.StartDag, "
                        "-9999 as EAGReeksWaardeID, "
                        "BakjeReeksWaarden.ID as BakjeReeksWaardeID, "
                        "-9999 as LaagReeksWaardeID FROM "
                        "(((((((EAG INNER JOIN Config ON Config.ID = EAG.ConfigID) "
                        "INNER JOIN GAF ON GAF.ID = Config.GAFID) "
                        "INNER JOIN Bakjes ON Bakjes.EAGID = EAG.ID) "
                        "INNER JOIN BakjeClusters ON BakjeClusters.BakjeID = Bakjes.ID) "
                        "INNER JOIN BakjeReeksWaarden ON BakjeReeksWaarden.BakjeClusterID = BakjeClusters.ID) "
                        "INNER JOIN BakjeClusterTypen on BakjeClusters.BakjeClusterTypeID = BakjeClusterTypen.ID) "
                        "INNER JOIN BakjeReeksen ON BakjeReeksen.ID = BakjeReeksWaarden.BakjeReeksID) "
                        "INNER JOIN BakjeTypen ON BakjeTypen.ID = Bakjes.BakjeTypeID "
                        "WHERE EAG.ID = {}".format(iEAG[0]))
    df1 = pd.read_sql(GETREEKSBAKJESQL, cnxn)

    GETREEKSLAAGSQL = ("SELECT EAG.ID AS EAGID, "
                        "EAG.Code AS EAGCode, "
                        "Bakjes.ID AS BakjeID, "
                        "LaagTypen.Laagvolgorde, "
                        "LaagClusterTypen.Code AS ClusterType, "
                        "LaagReeksen.Code AS ReeksType, "
                        "ReeksTypen.ReeksType as ParamType, "
                        "LaagReeksWaarden.Waarde AS Waarde, "
                        "LaagReeksWaarden.WaardeAlfa AS WaardeAlfa, "
                        "LaagReeksWaarden.StartDag, "
                        "-9999 as EAGReeksWaardeID, "
                        "-9999 as BakjeReeksWaardeID, "
                        "LaagReeksWaarden.ID as LaagReeksWaardeID FROM "
                        "(((((((((EAG INNER JOIN Config ON Config.ID = EAG.ConfigID) "
                        "INNER JOIN GAF ON GAF.ID = Config.GAFID) "
                        "INNER JOIN Bakjes ON Bakjes.EAGID = EAG.ID) "
                        "INNER JOIN Lagen ON Lagen.BakjeID = Bakjes.ID) "
                        "INNER JOIN LaagClusters ON LaagClusters.LaagID = Lagen.ID) "
                        "INNER JOIN LaagReeksWaarden ON LaagReeksWaarden.LaagClusterID = LaagClusters.ID) "
                        "INNER JOIN LaagClusterTypen on LaagClusters.LaagClusterTypeID = LaagClusterTypen.ID) "
                        "INNER JOIN LaagReeksen ON LaagReeksen.ID = LaagReeksWaarden.LaagReeksID) "
                        "INNER JOIN ReeksTypen ON ReeksTypen.ID = LaagReeksWaarden.ReeksTypeID)"
                        "INNER JOIN LaagTypen ON LaagTypen.ID = Lagen.LaagTypeID "
                        "WHERE EAG.ID = {}".format(iEAG[0]))
    df2 = pd.read_sql(GETREEKSLAAGSQL, cnxn)

    # TODO: test is this is what R function 'rbind' does
    df = pd.concat([df0, df1, df2], axis=0)

    if not df.empty:
        df.to_csv(os.path.join(outdir, "reeks_{0}_{1}.csv".format(iEAG[0], iEAG[1])), sep=";")

cnxn.close()
