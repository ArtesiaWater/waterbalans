"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans module.

"""

import waterbalans as wbs


# Create a polder instance
wb = wbs.Polder()

# Load the model structure from file or db
wb._load_model_data("data\\EAG_bakjes.csv", id="1000")
wb.load_series()
wb.load_parameters()


wb.calculate_wb()




