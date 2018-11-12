"""

Dit script bevat de automatische simulatie van waterbalansen op
EAG-niveau voor alle EAGs. De volgende drie invoerbestanden worden 
per EAG gebruikt:

- Modelstructuur (opp)
- Tijdreeksen
- Parameters

"""

import pandas as pd
import waterbalans as wb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.interactive(True)

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

f = open("info.log", "w")

# Loop over files
for i, (fbuckets, fparams, freeks) in file_df.iterrows():
    if not i.startswith("8030-EAG-2"):
        continue
    try:
        buckets = pd.read_csv(os.path.join(csvdir, fbuckets), delimiter=";",
                            decimal=",")
        buckets["OppWaarde"] = pd.to_numeric(buckets.OppWaarde)
        name = i
        eag_id = i.split("-")[-1]
        
        # Aanmaken van modelstructuur en de bakjes.
        e = wb.create_eag(eag_id, name, buckets)

        # Lees de tijdreeksen in
        reeksen = pd.read_csv(os.path.join(csvdir, freeks), delimiter=";",
                            decimal=",")
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
        ax.set_title(i)
        # Calculate and plot the chloride concentration
        C = e.calculate_chloride_concentration()
        C.plot()
    except Exception as e:
        print("Failed running {} because: '{}:{}'".format(i, type(e).__name__, e), file=f)

f.close()

plt.show()
