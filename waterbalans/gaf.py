"""This file contains the polder class

"""

from collections import OrderedDict
from io import BytesIO

import pandas as pd

from waterbalans.oud.io import load_model, read_xml
from waterbalans.oud.access import AccessServer
from .eag import Eag


class Gaf:
    """The Gaf class is the main object for a waterbalance.

    Parameters
    ----------
    id: int
        Integer id for the Gaf.

    Notes
    -----
    Gebieden-AfvoerGebieden

    """

    def __init__(self, id=None, db_model=None, db_series=None, db_param=None):
        self.id = id

        # Store the database source types
        self.db_model = db_model  # model database
        # self.db_series = self._connect_fews_db(args=db_series)
        self.db_param = self._connect_param_db(args=db_param)

        # Placeholder
        self.data = pd.DataFrame()
        self.eags = OrderedDict()
        self.parameters = pd.DataFrame(columns=["eag", "bucket", "pname",
                                                "pinit", "popt", "pmin",
                                                "pmax", "pvary"])

        self.series = OrderedDict()

    def add_eag(self, eag):
        self.eags[eag.name] = eag

    def _connect_fews_db(self, args):
        """Method tpo connect to the the FEWS database.

        Parameters
        ----------
        kwargs: dict
            Any keyword values combination that is taken by the FewsServer
            instance.

        Returns
        -------
        FewsServer: waterbalans.FewsServer

        """
        if args is None:
            args = {}
        return FewsServer(*args)

    def _connect_param_db(self, args):
        """Method to connect to a parameters database to extract the
        parameters.

        Parameters
        ----------
        kwargs:
            any parameters that are taken by the AccessServer object.

        Returns
        -------
        AccessServer: waterbalans.AccesServer

        """
        if args is None:
            args = {}
        return AccessServer(*args)

    def get_model_configs(self, gaf_id):
        model_configs = self.db_param.get_model_configs(gaf_id)
        return model_configs

    def get_model_sets(self, config_id):
        model_sets = self.db_param.get_model_sets(config_id)
        return model_sets

    def set_model_parameters(self, set_id):
        parameters = self.db_param.get_parameters(set_id)
        self.parameters._update_inplace(parameters)
        return parameters

    def get_model_structure(self, fname, id=None):
        """Method to load the data model from a file or database.

        Returns
        -------
        data: pandas.DataFrame
            Imported table that is used as a template to construct the model.

        """
        data = load_model(fname=fname)
        if id:
            self.data = data.loc[data.loc[:, "GAF"] == id]
        else:
            self.data = data

        self._create_model_structure()

    def _create_model_structure(self):
        """Method to import model structure description from a file or
        database.

        """
        for id in self.data.loc[:, "EAG"].unique():
            df = self.data.loc[self.data.loc[:, "EAG"] == id]
            eag = Eag(id=id, gaf=self, data=df, name=id)
            self.eags[id] = eag

    def get_series_list(self):
        """Method to obtain a list of measurement point id's to retrieve from
        the Fews server.

        Returns
        -------
        series: list
            List of time series that are available for this GAF.

        """
        pass

    def load_series(self):
        """Method to import the time series in both the Gaf object and
        the subpolder Objects.

        """
        self.series["prec"] = pd.Series()
        self.series["evap"] = pd.Series()

        for subpolder in self.eags.values():
            subpolder.load_series_from_eag()

    def get_series(self, id, **kwargs):
        kwargs.update(locationIds=id)
        data = self.db_series.getTimeSeries(**kwargs)
        data = BytesIO(data.encode())
        series = read_xml(data)

        return series

    def simulate(self):
        """Method to calculate the waterbalance for the Gaf.

        Returns
        -------

        """
        for eag in self.eags.values():
            print("Simulating the waterbalance for EAG: %s" % eag.name)
            eag.simulate()

    def validate(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.
        """
        pass
