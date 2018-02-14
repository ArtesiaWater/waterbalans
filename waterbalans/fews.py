"""This file contains the class to retrieve data from the FEWS database.

The server class is constructed according to the Singleton principle. This
means that only one instance of the server class, hence one server connection
is established. This prevents the creation of unlimited server connections.

"""
import pandas as pd
from zeep import Client

from waterbalans.utils import Singleton

WSDL = "http://localhost:8081/FewsPiService/fewspiservice?wsdl"
PARAMS = dict(
    locationIds="62023_stuw_1", parameterIds="Vol.berekend.dag",
    convertDatum=False,
    forecastSearchCount=1,
    importFromExternalDataSource=False,
    omitMissing=False, onlyHeaders=False,
    onlyManualEdits=False, showEnsembleMemberIds=False,
    showStatistics=False, showThresholds=False,
    useDisplayUnits=False,
    startForecastTime=pd.to_datetime("2017-01-01 00:00"))


class FewsServer(Client, metaclass=Singleton):
    def __init__(self, wsdl=WSDL, host="localhost", port=8081, strict=False):
        Client.__init__(self, wsdl=wsdl, strict=strict)

    def getTimeSeries(self, **kwargs):
        """Method to obtain a time series from the FEWS database. For now,
        this method only works to obtain a single time series.

        Parameters
        ----------
        kwargs: any valid argument to the fews server can be passed. Most
        important are probably the locationIds and the parameterIds.

        Returns
        -------
        series: pd.Series
            Pandas Series object with the requested time series.

        """
        parameters = PARAMS.copy()
        parameters.update(**kwargs)

        data = self.service.getTimeSeries(queryParams=parameters)

        return data
