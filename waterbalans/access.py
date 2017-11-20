"""This file contains the classes and methods used to connect to a microsoft
access database.

"""
import pyodbc

from waterbalans.utils import Singleton
from pandas import DataFrame
from numpy import nan


class AccessServer(metaclass=Singleton):
    __doc__ = """The access server 
    """

    def __init__(self, db_path="..\\..\\DB\\waterbalans.accdb",
                 db_driver="{Microsoft Access Driver (*.mdb, *.accdb)}"):
        self.db_path = db_path
        self.db_driver = db_driver

        self.connection = None
        self.connect()

    def connect(self, **kwargs):
        connect_string = (r'DRIVER=%s;' r'DBQ=%s;' % (self.db_driver,
                                                      self.db_path))
        self.connection = pyodbc.connect(connect_string, **kwargs)

    def close(self, **kwargs):
        """Method to close the connection with the Access Database.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        self.connection.close()
        self.connection = None

    def get_model_configs(self, gaf_id, default=True):
        """Method to obtain the model configurations for a gaf_id.

        Parameters
        ----------
        gaf_id
        default

        Returns
        -------

        """
        s = self.connection.execute("select ID from GAF where GafID=?", gaf_id)
        data = s.fetchall()
        s.close()
        data = [val[0] for val in data]  # Get rid of the tuples.
        return data

    def get_model_sets(self, config_id, default=True):
        """Method to obtain the modelsets for a specified GAF Id.

        Parameters
        ----------
        gaf_id

        Returns
        -------

        """
        s = self.connection.execute("select ID from ModelSet where "
                                    "ConfigID=?", config_id)
        data = s.fetchall()
        s.close()
        data = [val[0] for val in data]  # Get rid of the tuples.
        return data

    def get_parameters(self, set_id):
        """Method to obtain the parameters for a specified model set.

        Parameters
        ----------
        set

        Returns
        -------
        parameters: pandas.DataFrame
            returns a pandas dataframe with the parameter values

        """
        db_data = self.connection.execute("select Naam, Waarde, MaxWaarde, "
                                       "MinWaarde from ParamWaarden where "
                                       "ModelID=?", set_id)

        data = DataFrame(columns=["pinit", "pmin", "pmax", 'popt'])
        for row in db_data:
            data.loc[row.Naam, "pmin"] = row.MinWaarde
            data.loc[row.Naam, "pmax"] = row.MaxWaarde
            data.loc[row.Naam, "pinit"] = row.Waarde

        data["popt"] = data["pinit"]

        return data
