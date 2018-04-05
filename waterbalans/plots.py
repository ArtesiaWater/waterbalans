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


class Eag_Plots:
    def __init__(self, eag):
        self.eag = eag

    def bucket(self, name, freq="M"):
        bucket = self.eag.buckets[name]
        plotdata = bucket.fluxes.astype(float).resample(freq).mean()
        ax = plotdata.plot.bar(stacked=True, width=1)
        xticks = ax.axes.get_xticks()
        ax.set_xticks([i for i in range(0, 244, 12)])
        ax.set_xticklabels([i for i in range(1996, 2017, 1)])
