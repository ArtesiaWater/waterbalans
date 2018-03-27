"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans
module.

"""

import pandas as pd

import waterbalans as wb
from waterbalans.utils import excel2datetime

# Import some test time series
data = pd.read_csv("data\\2501_reeksen.csv", index_col="Date",
                   infer_datetime_format=True, dayfirst=True)
data.index = excel2datetime(data.index, freq="D")

# Create a polder instance
gaf = wb.Gaf()

series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
e = wb.Eag(gaf=gaf, name="2501-EAG-1", series=series)

# Water
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.087185 * 1e-3
series["w"] = 0.0
series["x"] = 3000.0 / 502900
b = wb.Water(id=1, eag=e, series=series, area=502900)
e.add_water(b)

# Verhard
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.0
b = wb.Verhard(id=2, eag=e, series=series, area=102894)
e.add_bucket(b)

# Gedraineerd
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.0
b = wb.Drain(id=3, eag=e, series=series, area=0)
e.add_bucket(b)

# Onverhard: >0 & <= 0.5 kwel
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.087185 * 1e-3
b = wb.Onverhard(id=4, eag=e, series=series, area=1840576)
e.add_bucket(b)

# Onverhard: >-0.4306 & <= 0 weinig wegzijging
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = -0.122533 * 1e-3
b = wb.Onverhard(id=5, eag=e, series=series, area=766779)
e.add_bucket(b)

# Onverhard: >-1 & <=-0.4306 meer wegzijging
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = -0.534395 * 1e-3
b = wb.Onverhard(id=6, eag=e, series=series, area=520862)
e.add_bucket(b)

# Onverhard: <=-1.26 veel wegzijging
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = -1.160258 * 1e-3
b = wb.Onverhard(id=7, eag=e, series=series, area=0)
e.add_bucket(b)

gaf.add_eag(e)

params = pd.read_excel("data\parameters.xlsx", sheet_name="2501-01-F001")

e.simulate(parameters=params)

# wb.simulate()


# SELECT GAF.ID, Config.ID, EAG.ID, Bakjes.ID, Bakjes.BakjeTypeID
#
# FROM ((GAF INNER JOIN Config ON GAF.ID = Config.GafID) INNER JOIN EAG ON
# Config.ID = EAG.ConfigID) INNER JOIN Bakjes ON EAG.ID = Bakjes.EAGID;


d = {
    "q_oa_2": "verhard",
    "p": "neerslag",
    "e": "verdamping",
    "s": "kwel",
    "w": "wegzijging",

    "x": "maalstaat",
}

fluxes = e.water.fluxes.loc[:, "q_oa_2"]
