#%%
import pandas as pd
import waterbalans as wb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re

# %%
mpl.interactive(True)

# time start
t0 = pd.datetime.now()

# Set database url connection
wb.pi.setClient(wsdl='http://localhost:8081/FewsPiService/fewspiservice?wsdl')

# Get all files
csvdir = r"C:/Users\dbrak/Documents/01-Projects/17026004_WATERNET_Waterbalansen/03data/DataExport_frompython"
files = os.listdir(csvdir)
eag_df = pd.DataFrame(data=files, columns=["filenames"])
eag_df["ID"] = eag_df.filenames.apply(lambda s: s.split("_")[2].split(".")[0])
eag_df["type"] = eag_df.filenames.apply(lambda s: s.split("_")[0])
eag_df.drop_duplicates(subset=["ID", "type"], keep="first", inplace=True)
file_df = eag_df.pivot(index="ID", columns="type", values="filenames")
file_df.dropna(how="any", axis=0, inplace=True)

# these EAGs provide output:
successful_eags = [
    "2140-EAG-3",
    "2250-EAG-2",
    "2500-EAG-6",
    "2501-EAG-1",
    "2501-EAG-2",
    "2505-EAG-1",
    "2510-EAG-2",
    "2510-EAG-3",
    "7060-EAG-1",
    "8030-EAG-2"
    ]

eagcode = successful_eags[9]

# Loop over files
fbuckets, fparams, freeks = file_df.loc[eagcode]

try:
    buckets = pd.read_csv(os.path.join(csvdir, fbuckets), delimiter=";",
                        decimal=",")
    buckets["OppWaarde"] = pd.to_numeric(buckets.OppWaarde)
    name = eagcode
    eag_id = eagcode.split("-")[-1]

    # Aanmaken van modelstructuur en de bakjes.
    e = wb.create_eag(eag_id, name, buckets)

    # Lees de tijdreeksen in
    reeksen = pd.read_csv(os.path.join(csvdir, freeks), delimiter=";",
                        decimal=",")

    # temporary shift in timeseries to match excel sheets!
    # reeksen.index = reeksen.index - pd.Timedelta(days=1)
    e.add_series(reeksen)

    # Simuleer de waterbalans
    params = pd.read_csv(os.path.join(csvdir, fparams), delimiter=";",
                            decimal=",")
    params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)

    e.simulate(params=params, tmin="2000", tmax="2015-12-31")

    # Calculate and plot the fluxes as a bar plot
    fluxes = e.aggregate_fluxes()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    fluxes.loc["2010":"2015"].resample("M").mean().plot.bar(stacked=True, width=1, ax=ax)
    ax.set_title(eagcode)
    
    # Calculate and plot the chloride concentration
    # C = e.calculate_chloride_concentration()
    # fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    # C.plot(ax=ax2, label=e.name)
    # ax2.grid(b=True)
    # ax2.legend(loc="best")

except Exception as e:
    print("Failed running {} because: '{}:{}'".format(eagcode, type(e).__name__, e))


# Water balance closed?
b = e.water.validate()
wb = e.water.validate(return_wb_series=True)
print(e.name, "Water balance closed: ", b)

# Get EAG reeksen from Excel:
eagcode_excel = e.name + "_F001"  # Manual entry of version for now

lookup = pd.read_excel(r"C:\Users\dbrak\Documents\01-Projects\17026004_WATERNET_Waterbalansen\03data\excel_balansen\lookup_Excel.xlsx",
                       skiprows=15, header=0, index_col=None)
excel_eag = r"C:\Users\dbrak\Documents\01-Projects\17026004_WATERNET_Waterbalansen\03data\excel_balansen\{}.xlsx".format(eagcode_excel)

eag_dict = {}
for i, irow in lookup.iterrows():
    s = re.split('(\d+)', irow.Range)
    if len(s) > 3:
        c0, r0, c1, r1, _ = s
        nrows = np.int(r1) - np.int(r0)
        usecols = "A," + c0
        index = [0]
    else:
        c0, r0, _ = s
        continue
    ts = pd.read_excel(excel_eag, skiprows=np.int(r0)-1, header=None, 
                                         index=index, usecols=usecols, sheet_name=irow.Sheet,
                                         nrows=nrows)
    if len(ts.columns) > 1:
        print("setting index")
        ts.set_index(0, inplace=True)
    
    eag_dict[irow.Reeks] = ts

# for k in eag_dict.keys():
#     r = eag_dict[k]
#     r = pd.to_numeric(r.iloc[:, 0], errors="coerce")
#     eag_dict[k] = r.loc[r.index.dropna()]

# get dates for input
r0 = np.int(re.split('(\d+)', lookup.Range.iloc[0])[1])
r1 = np.int(re.split('(\d+)', lookup.Range.iloc[0])[3])
nrows = np.int(r1) - np.int(r0)
eag_dict["datetime"] = pd.read_excel(excel_eag, skiprows=r0-1, header=None, nrows=nrows,
                                     index=None, usecols="A", sheet_name="uitgangspunten")

# get dates for bakje # 1: verhard
r0 = np.int(re.split('(\d+)', lookup.Range.iloc[2])[1])
r1 = np.int(re.split('(\d+)', lookup.Range.iloc[2])[3])
nrows = np.int(r1) - np.int(r0)
eag_dict["datetime_bakje"] = pd.read_excel(excel_eag, skiprows=r0-1, header=None, nrows=nrows,
                                           index=None, usecols="A", sheet_name="Rekenblad")

# Plot comparison
# Neerslag + Verdamping
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), dpi=150, sharex=True)
ax0.plot(eag_dict["neerslag"].index, eag_dict["neerslag"], label="neerslag Excel")
# ax0.plot(eag_dict["neerslag"], label="neerslag Excel")
ax0.plot(e.series.Neerslag.index, e.series.Neerslag*1e3, label="neerslag FEWS")
ax0.legend(loc="best")

ax1.plot(eag_dict["verdamping"].index, eag_dict["verdamping"], label="verdamping Excel")
# ax1.plot(eag_dict["verdamping"], label="verdamping Excel")
ax1.plot(e.series.Verdamping.index, e.series.Verdamping*1e3, label="verdamping FEWS")
ax1.legend(loc="best")


# Bakje reeksen
if e.buckets[buckets.BakjeID.iloc[0]].name == "Verhard":
    b1 = e.buckets[buckets.BakjeID.iloc[0]]
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150, sharex=True)

    # neerslagoverschot
    ax0.plot(eag_dict["neerslagoverschot"].index, eag_dict["neerslagoverschot"], 
            label="neerslagoverschot Excel")
    ax0.plot(b1.fluxes.index, b1.fluxes.loc[:, "q_no"]*b1.area, label="neerslagoverschot Python")

    # kwel
    ax1.plot(eag_dict["Seepage"].index, eag_dict["Seepage"], 
            label="Seepage Excel")
    ax1.plot(b1.fluxes.index, b1.fluxes.loc[:, "q_s"]*b1.area, label="Seepage Python")

    # intrek/uitspoeling
    ax2.plot(eag_dict["Intrek/Uitspoeling"].index, eag_dict["Intrek/Uitspoeling"], 
            label="Intrek/Uitspoeling Excel")
    ax2.plot(b1.fluxes.index, b1.fluxes.loc[:, "q_ui"]*b1.area, label="Intrek/Uitspoeling Python")

    # oppervlakkige afstroming
    ax3.plot(eag_dict["Opp. Afstroming"].index, eag_dict["Opp. Afstroming"], 
            label="Opp. Afstroming Excel")
    ax3.plot(b1.fluxes.index, b1.fluxes.loc[:, "q_oa"]*b1.area, label="Opp. Afstroming Python")

    for iax in [ax0, ax1, ax2, ax3]:
        iax.grid(b=True)
        iax.legend(loc="best")
    
    fig.suptitle("Bakje Verhard")

plt.show()

print("Time elapsed: {0:.2f} minutes".format((pd.datetime.now() - t0).total_seconds()/60.))
