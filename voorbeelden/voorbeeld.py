"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans
module.

"""

import waterbalans as wbs

# Create a polder instance
wb = wbs.Polder()

# Laad de parameters in voor een bepaalde gaf en model
configs = wb.get_model_configs(gaf_id=2501)     # Levert een lijst met id's op.
sets = wb.get_model_sets(config_id=configs[0])  # Levert een lijst met id's op.
wb.set_model_parameters(set_id=sets[0])         # Laadt de parameters in.

# Laad de model structuur in vanuit GIS
wb._load_model_data("data\\EAG_bakjes.csv", id="2501")

wb.get_model_structure()


# Laad de tijdreeksen in uit FEWS
wb.get_series("60010_pomp_3")

# Load the model structure from file or db

wb.load_series()



"66003_Verdamping"
"66003_Neerslag"

names = ["60010_pomp_3", "60038_Benedenstrooms_stuw_1",
         "60038_Bovenstrooms_stuw_1"]
for name in names:
    wb.get_series(name).plot()
