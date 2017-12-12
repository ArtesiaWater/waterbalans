"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans
module.

"""

import pandas as pd

import waterbalans as wbs

# Create a polder instance
wb = wbs.Polder()

# Laad de parameters in voor een bepaalde gaf en model
configs = wb.get_model_configs(gaf_id=2501)  # Levert een lijst met id's op.
sets = wb.get_model_sets(config_id=configs[0])  # Levert een lijst met id's op.
wb.set_model_parameters(set_id=sets[0])  # Laadt de parameters in.

# Laad de model structuur in vanuit GIS
wb.get_model_structure("data\\EAG_bakjes.csv", id="2501")

# Laad de tijdreeksen in uit FEWS, DOET NU NIKS!!!
wb.load_series()

#
# wb.get_series("60010_pomp_3")
#
# names = ["60010_pomp_3", "60038_Benedenstrooms_stuw_1",
#          "60038_Bovenstrooms_stuw_1", "66003_Neerslag", "66003_Verdamping"]
# for name in names:
#     wb.get_series(name).plot()


# Test de Onverhard Bucket
from waterbalans.buckets import Drain, Onverhard, Verhard
from waterbalans.utils import excel2datetime
import matplotlib.pyplot as plt

data = pd.read_csv("data\\2501_reeksen.csv", index_col="Date",
                   infer_datetime_format=True, dayfirst=True)
data.index = excel2datetime(data.index, freq="D")

b = Onverhard(wb, data)
b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
b.series["s"] = 0.087185 * 1e-3
b.parameters.loc[["v_eq", "v_max", "fmin", "fmax", "i_fac", "u_fac", "n"],
                 "optimal"] = 0.0, 0.6, 0.75, 1, 0.05, 0.1, 0.1
b.calculate_wb()
area = 1840576.0

# Make plot to compare ro excel
plotdata = (b.fluxes * area).astype(float).resample("M").mean()
ax = plotdata.loc[:, ["q_s", "q_no", "q_ui", "q_oa"]].plot.bar(
    stacked=True,
                                                           width=1,
                                                           color=["r", "b",
                                                                  "lightgreen",
                                                                  "darkgreen"])
xticks = ax.axes.get_xticks()
ax.set_xticks([i for i in range(0, 244, 12)])
ax.set_xticklabels([i for i in range(1996, 2017, 1)])
plt.legend(["kwel/wegz", "opp afsp.", "uitspl/intrek", "neerslagoverschot"])


# b = Verhard(wb, data, area=102894.0)
# b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
# b.series["s"] = 0.0
# b.parameters.loc[["v_eq", "v_max1", "v_max2", "fmin", "fmax", "i_fac",
#                   "u_fac", "n1", "n2"],
#                  "optimal"] = 0.0, 0.002, 1.0, 1.0, 1.0, 0.10, 0.10, 1, 0.2
# b.calculate_wb()
# area = 102894.0

b = Drain(wb, data, area=1000)
b.series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
b.series["s"] = 0.09 * 1e-3
b.parameters.loc[["v_eq", "v_max1", "v_max2", "fmin", "fmax", "i_fac",
                  "u_fac1", "u_fac2", "n1", "n2"],
                 "optimal"] = 0.0, 0.7, 0.30, 0.75, 1.0, 0.001, 0.10, 0.001, \
                              0.3, 0.3
b.calculate_wb()
area = 1000

# Make plot to compare ro excel
plotdata = (b.fluxes * area).astype(float).resample("M").mean()
ax = plotdata.loc[:, ["q_s", "q_no", "q_ui", "q_dr"]].plot.bar(
    stacked=True,
    width=1,
    color=["r", "b",
           "lightgreen",
           "darkgreen"])
xticks = ax.axes.get_xticks()
ax.set_xticks([i for i in range(0, 244, 12)])
ax.set_xticklabels([i for i in range(1996, 2017, 1)])
plt.legend(["kwel/wegz", "opp afsp.", "uitspl/intrek", "drain"])
