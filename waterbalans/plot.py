"""The Plot module contains the methods for plotting of a waterbalans.

Author: R.A. Collenteur, Artesia Water, 2017-11-20

"""

class Plot():
    def __init__(self, waterbalans):
        self.wb = waterbalans

    def series(self):
        """Method to plot all series that are present in the model

        Returns
        -------
        axes: list of matplotlib.axes
            Liost with matplotlib axes instances.

        """

        raise NotImplementedError
