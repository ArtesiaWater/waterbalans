# Test script voor FEWS voor Ben Staring - 2018-02-01

import waterbalans as wbs

# Create a polder instance
wb = wbs.Gaf()

wb.get_series("60010_pomp_3")

names = ["60010_pomp_3", "60038_Benedenstrooms_stuw_1",
         "60038_Bovenstrooms_stuw_1", "66003_Neerslag", "66003_Verdamping"]
for name in names:
    wb.get_series(name).plot()
