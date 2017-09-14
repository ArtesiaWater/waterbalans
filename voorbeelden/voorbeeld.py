"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans module.

"""

import waterbalans as wbs

wb = wbs.Polder(id="test", db_model="json")



wb.calculate_wb()




