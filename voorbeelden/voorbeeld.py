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

# Onverhard
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.087185 * 1e-3
b = wb.Onverhard(eag=e, series=series, area=1656518)
b.parameters.loc[["v_eq", "v_max", "fmin", "fmax", "i_fac", "u_fac", "n"],
                 "optimal"] = 0.0, 0.6, 0.75, 1, 0.05, 0.1, 0.1
e.add_bucket(b)

# Verhard
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.0
b = wb.Verhard(e, series=series, area=43547)
b.parameters.loc[["v_eq", "v_max1", "v_max2", "fmin", "fmax", "i_fac", "u_fac",
                  "n1", "n2"], "optimal"] = 0.0, 0.002, 1.0, 1.0, 1.0, \
                                            0.10, 0.10, 1, 0.2
e.add_bucket(b)

# Gedraineerd
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.0
b = wb.Drain(e, series=series, area=184058)
b.parameters.loc[["v_eq", "v_max1", "v_max2", "fmin", "fmax", "i_fac",
                  "u_fac1", "u_fac2", "n1", "n2"], "optimal"] = 0.0, 0.7, 0.30, \
                                                                0.75, 1.0, \
                                                                0.001, 0.10, \
                                                                0.001, 0.3, 0.3
e.add_bucket(b)

# Water
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
series["s"] = 0.087185 * 1e-3
series["w"] = -3000.0 / 293251
b = wb.Water(e, series, area=293251)
b.parameters.loc[["h_eq", "h_min", "h_max", "q_max"],
                 "optimal"] = -2.46, -2.47, -2.45, 369672.0
e.add_bucket(b)

gaf.add_eag(e)
gaf.simulate()


# # Laad de parameters in voor een bepaalde gaf en model
# configs = wb.get_model_configs(gaf_id=2501)  # Levert een lijst met id's op.
# sets = wb.get_model_sets(config_id=configs[0])  # Levert een lijst met id's op.
# wb.set_model_parameters(set_id=sets[0])  # Laadt de parameters in.
#
# # Laad de model structuur in vanuit GIS
# wb.get_model_structure("data\\EAG_bakjes.csv", id="2501")

#wb.simulate()
