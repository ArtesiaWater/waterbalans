from datetime import datetime

import pandas as pd

from hkvfewspy.io.fewspi import pi

startTime = datetime(2017, 12, 29)
endTime = datetime(2018, 1, 1)

pi.setClient(wsdl='http://localhost:8081/FewsPiService/fewspiservice?wsdl')

lijst = pd.read_csv("waterbalans.txt", sep=";", header=None,
                    names=["filterId", "moduleInstanceId", "locationId",
                           "parameterId", "nanValue", "name"])

warning = {}

for _, row in lijst.head().iterrows():
    moduleInstanceId, parameterId, locationId = row.loc[["moduleInstanceId",
                                                         "parameterId",
                                                         "locationId"]]

    params = dict(
        moduleInstanceIds=[moduleInstanceId],
        parameterIds=[parameterId],
        locationIds=[locationId],
        startTime=datetime(2015, 12, 29),
        endTime=endTime,
        clientTimeZone='Utc/GMT+1',
        forecastSearchCount=1,
        convertDatum='false',
        useDisplayUnits='false',
        showThresholds='true',
        omitMissing='false',
        onlyHeaders='false',
        onlyManualEdits='false',
        showStatistics='false',
        ensembleId='',
        importFromExternalDataSource='false',
        showEnsembleMemberIds='false',
        version='1.22'
    )
    try:
        df, entry = pi.getTimeSeries(params, setFormat='df')
        df.reset_index(inplace=True)
        df = df.loc[:, ["date", "value"]].set_index("date")
        df.value.plot()
    except:
        warning[locationId] = [moduleInstanceId, parameterId, locationId]

