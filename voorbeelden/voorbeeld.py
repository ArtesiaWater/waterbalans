"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans
module.

"""

import matplotlib.pyplot as plt
import pandas as pd

import waterbalans as wbs
from waterbalans.buckets import Drain, Onverhard, Verhard, Water
from waterbalans.utils import excel2datetime

# Create a polder instance
wb = wbs.Gaf()

# Laad de parameters in voor een bepaalde gaf en model
configs = wb.get_model_configs(gaf_id=2501)  # Levert een lijst met id's op.
sets = wb.get_model_sets(config_id=configs[0])  # Levert een lijst met id's op.
wb.set_model_parameters(set_id=sets[0])  # Laadt de parameters in.

# Laad de model structuur in vanuit GIS
wb.get_model_structure("data\\EAG_bakjes.csv", id="2501")

# Test de Onverhard Bucket
data = pd.read_csv("data\\2501_reeksen.csv", index_col="Date",
                   infer_datetime_format=True, dayfirst=True)
data.index = excel2datetime(data.index, freq="D")

area = 1656518
b = Onverhard(wb, data, area=area)
b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
b.series["s"] = 0.087185 * 1e-3
b.parameters.loc[["v_eq", "v_max", "fmin", "fmax", "i_fac", "u_fac", "n"],
                 "optimal"] = 0.0, 0.6, 0.75, 1, 0.05, 0.1, 0.1
b.calculate_wb()
wb.subpolders["2501-EAG-1"].buckets["Onverhard"] = b

area = 43547
b = Verhard(wb, data, area=area)
b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
b.series["s"] = 0.0
b.parameters.loc[["v_eq", "v_max1", "v_max2", "fmin", "fmax", "i_fac", "u_fac",
                  "n1", "n2"], "optimal"] = 0.0, 0.002, 1.0, 1.0, 1.0, \
                                            0.10, 0.10, 1, 0.2
b.calculate_wb()
wb.subpolders["2501-EAG-1"].buckets["Verhard"] = b

area = 184058
b = Drain(wb, data, area=area)
b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
b.series["s"] = 0.0 * 1e-3
b.parameters.loc[["v_eq", "v_max1", "v_max2", "fmin", "fmax", "i_fac",
                  "u_fac1", "u_fac2", "n1", "n2"], "optimal"] = 0.0, 0.7, 0.30, \
                                                                0.75, 1.0, \
                                                                0.001, 0.10, \
                                                                0.001, 0.3, 0.3
b.calculate_wb()
wb.subpolders["2501-EAG-1"].buckets["Drain"] = b

# Open Water bakje
area = 293251
b = Water(wb.subpolders["2501-EAG-1"], data, area=area)
b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
b.series["s"] = 0.087185 * 1e-3
b.series["w"] = -3000.0 / area
b.parameters.loc[["h_eq", "h_min", "h_max", "q_max"],
                 "optimal"] = -2.46, -2.47, -2.45, 369672.0
b.calculate_wb()


def plot_bucket(bucket):
    plotdata = bucket.fluxes.astype(float).resample("M").mean()
    ax = plotdata.plot.bar(stacked=True, width=1)
    xticks = ax.axes.get_xticks()
    ax.set_xticks([i for i in range(0, 244, 12)])
    ax.set_xticklabels([i for i in range(1996, 2017, 1)])
