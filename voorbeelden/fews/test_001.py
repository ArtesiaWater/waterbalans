from datetime import datetime

import pandas as pd

from hkvfewspy import Pi

startTime = datetime(2017, 1, 29)
endTime = datetime(2018, 6, 1)

pi = Pi()
pi.setClient(wsdl='http://localhost:8081/FewsPiService/fewspiservice?wsdl')

lijst = pd.read_csv("./voorbeelden/fews/waterbalans.txt", sep=";", header=None,
                    names=["filterId", "moduleInstanceId", "locationId",
                           "parameterId", "nanValue", "name"])

warning = {}

for _, row in lijst.head().iterrows():
    moduleInstanceId, parameterId, locationId = row.loc[["moduleInstanceId",
                                                         "parameterId",
                                                         "locationId"]]

    query = pi.setQueryParameters(prefill_defaults=True)
    query.query["onlyManualEdits"] = False
    query.query["qualifierId"] = ["min"]
    query.parameterIds([parameterId])
    query.moduleInstanceIds([moduleInstanceId])
    query.locationIds([locationId])
    query.startTime(startTime)
    query.endTime(endTime)
    query.clientTimeZone('Europe/Amsterdam')

    try:
        df = pi.getTimeSeries(query, setFormat='df')
        df.reset_index(inplace=True)
        df = df.loc[:, ["date", "value"]].set_index("date")
        df.value.plot()
    except:
        warning[locationId] = [moduleInstanceId, parameterId, locationId]
