import waterbalans as wb

# Time options
tmin = "1996"
tmax = "2015"

# Get ModelStructure, TimeSeries, and Parameters from example excel file
df_ms, df_ts, df_params = wb.utils.get_model_input_from_excel(
    r"./data/example_input_for_waterbalance.xlsx")

# Get ID and Name
eag_id = df_ms.loc[0, "EAGID"]
eag_name = df_ms.loc[0, "EAGCode"]

# Create EAG
e = wb.create_eag(eag_id, eag_name, df_ms, use_waterlevel_series=False)

# Add TimeSeries
e.add_series(df_ts, tmin=tmin, tmax=tmax)

# Add extra series (optional)
df_series = wb.utils.get_extra_series_from_excel(
    r"./data/example_input_for_waterbalance.xlsx")
wb.utils.add_timeseries_to_eag(e, df_series, tmin=tmin,
                               tmax=tmax, overwrite=False)

# Simulate
e.simulate(df_params, tmin=tmin, tmax=tmax)

# Plot aggregated fluxes
e.plot.aggregated(tmin=tmin, tmax=tmax)
