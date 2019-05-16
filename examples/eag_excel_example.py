import matplotlib.pyplot as plt
import waterbalans as wb

# Time options
tmin = "1996"
tmax = "2015"

# %% WATER QUANTITY

# Get ModelStructure, TimeSeries, and Parameters from example excel file
excelfile = r"./data/example_input_for_waterbalance.xlsx"
df_ms, df_ts, df_params = wb.utils.get_model_input_from_excel(excelfile)

# Get ID and Name
eag_id = df_ms.loc[0, "EAGID"]
eag_name = df_ms.loc[0, "EAGCode"]

# Create EAG
e = wb.create_eag(eag_id, eag_name, df_ms, use_waterlevel_series=False)

# Add TimeSeries
e.add_series_from_database(df_ts, tmin=tmin, tmax=tmax)

# Add extra series (optional)
df_series = wb.utils.get_extra_series_from_excel(excelfile)
wb.utils.add_timeseries_to_obj(e, df_series, tmin=tmin,
                               tmax=tmax, overwrite=False)

# Simulate
e.simulate(df_params, tmin=tmin, tmax=tmax)

# Plot aggregated fluxes
e.plot.aggregated(tmin=tmin, tmax=tmax)

# %% WATER QUALITY

# Get data from excelfile
wq_params = wb.utils.get_wqparams_from_excel(excelfile)

chloride = e.simulate_wq(wq_params, increment=False, tmin=tmin, tmax=tmax)

e.plot.chloride(chloride, tmin=tmin, tmax=tmax)

plt.show()
