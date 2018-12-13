"""

Dit voorbeeld bevat de automatische simulatie van een waterbalans op
EAG-niveau. De volgende drie invoerbestanden worden gebruikt:

- Modelstructuur
- Tijdreeksen
- Parameters

"""

import pandas as pd
import waterbalans as wb

# Set database url connection
wb.pi.setClient(wsdl='http://localhost:8081/FewsPiService/fewspiservice?wsdl')

buckets = pd.read_csv("data\\opp_19578_2501-EAG-1.csv", delimiter=";",
                      decimal=",")
name = "2501-EAG-01"
id = 1
# Aanmaken van modelstructuur en de bakjes.
e = wb.create_eag(id, name, buckets)

# Lees de tijdreeksen in
reeksen = pd.read_csv("data\\reeks_19578_2501-EAG-1.csv", delimiter=";",
                      decimal=",")
e.add_series(reeksen)

# Simuleer de waterbalans
params = pd.read_csv("data\\param_19578_2501-EAG-1.csv", delimiter=";",
                     decimal=",")
params.rename(columns={"ParamCode": "Code"}, inplace=True)
params["Waarde"] = pd.to_numeric(params.Waarde)

e.simulate(params=params, tmin="2000", tmax="2015-12-31")

# Calculate and plot the fluxes as a bar plot
fluxes = e.aggregate_fluxes()
fluxes.loc["2010":"2015"].resample("M").mean().plot.bar(stacked=True, width=1)

# Calculate and plot the chloride concentration
C = e.calculate_chloride_concentration()
C.plot()
