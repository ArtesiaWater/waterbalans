"""The Plot module contains the methods for plotting of a waterbalans.

Author: R.A. Collenteur, Artesia Water, 2017-11-20

"""
import numpy as np
from pandas import Timestamp
import matplotlib.pyplot as plt
from matplotlib import colors
from .timeseries import get_series


class Plot():
    def __init__(self, waterbalans):
        self.wb = waterbalans
        self.colordict =  {"kwel": "brown",
                           "neerslag": "b",                             
                           "uitspoeling": "lime", 
                           "drain": "orange",
                           "verhard": "darkgray",
                           "afstroming": "darkgreen", 
                           "q_cso": "olive", 
                           "wegzijging": "brown", 
                           "verdamping": "b", 
                           "intrek": "lime",
                           "uitlaat": "salmon", 
                           "berekende inlaat": "r", 
                           "berekende uitlaat": "k"}

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
        self.colordict =  {"kwel": "brown",
                           "neerslag": "b", 
                           "uitspoeling": "lime",  
                           "drain": "orange",
                           "verhard": "darkgray",
                           "afstroming": "darkgreen", 
                           "q_cso": "olive", 
                           "wegzijging": "brown", 
                           "verdamping": "b", 
                           "intrek": "lime", 
                           "uitlaat": "salmon",
                           "berekende inlaat": "r", 
                           "berekende uitlaat": "k"}

    def bucket(self, name, freq="M", tmin=None, tmax=None):
        bucket = self.eag.buckets[name]

        # get tmin, tmax if not defined
        if tmin is None:
            bucket.fluxes.index[0]
        if tmax is None:
            bucket.fluxes.index[-1]

        # get data
        plotdata = bucket.fluxes.loc[tmin:tmax].astype(float).resample(freq).mean()
        
        # get correct colors per flux
        rgbcolors = []
        i = 0
        for icol in plotdata.columns:
            if icol in self.colordict.keys():
                rgbcolors.append(colors.to_rgba(self.colordict[icol]))
            else:
                # if no color defined get one of the default ones
                rgbcolors.append(colors.to_rgba("C{}".format(i%10)))
                i += 1
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax = plotdata.plot.bar(stacked=True, width=1, color=rgbcolors, ax=ax)
        
        ax.set_title("{}: bakje {}".format(self.eag.name, bucket.name))
        ax.set_ylabel("<- uitstroming | instroming ->")
        ax.grid(axis="y")
        
        ax.legend(ncol=2)

        # set ticks (currently only correct for monthly data)
        if freq == "M":
            ax.set_xticklabels([dt.strftime('%b-%y') for dt in 
                                plotdata.index.to_pydatetime()])
        fig.tight_layout()
        
        return ax
    
    def aggregated(self, freq="M", tmin=None, tmax=None):
        fluxes = self.eag.aggregate_fluxes()

        # get tmin, tmax if not defined
        if tmin is None:
            fluxes.index[0]
        if tmax is None:
            fluxes.index[-1]

        # get data and sort columns
        plotdata = fluxes.loc[tmin:tmax].astype(float).resample(freq).mean()
        column_order = [k for k in self.colordict.keys()]
        plotdata = plotdata.loc[:, column_order]

        # get correct colors per flux
        rgbcolors = []
        i = 0
        for icol in plotdata.columns:
            if icol in self.colordict.keys():
                rgbcolors.append(colors.to_rgba(self.colordict[icol]))
            else:
                # if no color defined get one of the default ones
                rgbcolors.append(colors.to_rgba("C{}".format(i%10)))
                i += 1
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax = plotdata.plot.bar(stacked=True, width=1, color=rgbcolors, ax=ax)
        
        ax.set_title(self.eag.name)
        ax.set_ylabel("<- uitstroming | instroming ->")
        ax.grid(axis="y")
        
        ax.legend(ncol=2)

        # set ticks (currently only correct for monthly data)
        if freq == "M":
            ax.set_xticklabels([dt.strftime('%b-%y') for dt in 
                                plotdata.index.to_pydatetime()])
        fig.tight_layout()
        
        return ax

    def gemaal(self, tmin=None, tmax=None):
        fluxes = self.eag.aggregate_fluxes()
        
        # get tmin, tmax if not defined
        if tmin is None:
            fluxes.index[0]
        if tmax is None:
            fluxes.index[-1]
        
        fluxes = fluxes.loc[tmin:tmax]
        calculated_out = fluxes.loc[:, ["berekende uitlaat"]]
        
        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.plot(calculated_out.index, -1*calculated_out, lw=2, label="berekende uitlaat")
        
        if "Gemaal" in self.eag.series.columns:
            gemaal = self.eag.series["Gemaal"].loc[tmin:tmax]
            ax.plot(gemaal.index, gemaal, lw=2, label="gemeten bij gemaal")

        ax.set_ylabel("Afvoer (m$^3$/dag)")
        ax.legend(loc="best")
        ax.grid(b=True)
        fig.tight_layout()

        return ax

    def gemaal_cumsum(self, tmin=None, tmax=None, period="year", inlaat=True):
        fluxes = self.eag.aggregate_fluxes()
        
        # get tmin, tmax if not defined
        if tmin is None:
            fluxes.index[0]
        if tmax is None:
            fluxes.index[-1]
        
        fluxes = fluxes.loc[tmin:tmax]
        calculated_out = fluxes.loc[:, ["berekende uitlaat"]]

        if period == "year":
            grouper = [fluxes.index.year]
        elif period == "month":
            grouper = [fluxes.index.year, fluxes.index.month]
        else:
            grouper=None

        if grouper is not None:
            calculated_out = calculated_out.groupby(by=grouper).cumsum()
            if inlaat:
                calculated_in = fluxes.loc[:, ["berekende inlaat"]].groupby(by=grouper).cumsum()
        else:
            calculated_out = calculated_out.cumsum()
            if inlaat:
                calculated_in = fluxes.loc[:, ["berekende inlaat"]].cumsum()
        
        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.plot(calculated_out.index, calculated_out, lw=2, label="berekende uitlaat")
        if inlaat:
            ax.plot(calculated_in.index, calculated_in, lw=2, label="berekende inlaat", color="red")
        
        if "Gemaal" in self.eag.series.columns:
            gemaal = self.eag.series["Gemaal"].loc[tmin:tmax]
            if period == "year":
                gemaal = gemaal.groupby(by=gemaal.index.year).cumsum()
            elif period == "month":
                gemaal = gemaal.groupby(by=[gemaal.index.year, gemaal.index.month]).cumsum()
            else:
                gemaal = gemaal.cumsum()
            ax.fill_between(gemaal.index, 0.0, -gemaal, label="gemeten bij gemaal", color="C1")

        ax.set_ylabel("Cumulatieve afvoer (m$^3$)")
        ax.legend(loc="best")
        ax.grid(b=True)
        fig.tight_layout()

        return ax

    def chloride(self, c, tmin=None, tmax=None):
        # get tmin, tmax if not defined
        if tmin is None:
            self.eag.series.index[0]
        if tmax is None:
            self.eag.series.index[-1]
        
        # TODO: make plot function indepent of passing concentrations by adding chloride to eag object
        c = c.loc[tmin:tmax]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.plot(c.index, c, label=self.eag.name)
        ax.grid(b=True)
        ax.legend(loc="best")
        ax.set_ylabel("Chloride concentratie (mg/L)")

        return ax
    
    def chloride_fractions(self, tmin=None, tmax=None, chloride_conc=True):
        # get tmin, tmax if not defined
        if tmin is None:
            self.eag.series.index[0]
        if tmax is None:
            self.eag.series.index[-1]

        colordict = {"kwel": "brown", 
                     "neerslag": "blue", 
                     "uitspoeling": "lime", 
                     "afstroming": "darkgreen", 
                     "drain": "orange", 
                     "berekende inlaat": "red", 
                     "q_cso": "black",
                     "verhard": "gray"} 
        
        fr = self.eag.calculate_fractions().loc[tmin:tmax]

        fr_list = [fr["initial"].astype(np.float).values]
        labels = ["initieel"]
        colors = ["lightgray"]
        for icol, c in colordict.items():
            if icol in fr.columns:
                fr_list.append(fr[icol].astype(np.float).values)
                colors.append(c)
                labels.append(icol)
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.stackplot(fr.index, *fr_list, labels=labels, colors=colors)
        ax.grid(b=True)
        ax.legend(loc="upper center", ncol=2, )
        ax.set_ylabel("Percentage (%)")

        if chloride_conc:
            C = self.eag.calculate_chloride_concentration()
            ax2 = ax.twinx()
            ax2.plot(C.index, C, color="k")
            ax2.set_ylabel("Chloride concentration (mg/L)")
        ax.set_xlim(Timestamp(tmin), Timestamp(tmax))
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return ax

    def water_level(self, label_obs=False):

        hTarget = self.eag.water.parameters.loc["hTarget_1", "Waarde"]
        hTargetMax = hTarget + self.eag.water.parameters.loc["hTargetMax_1", "Waarde"]
        hTargetMin = hTarget - self.eag.water.parameters.loc["hTargetMin_1", "Waarde"]
        hBottom = self.eag.water.parameters.loc["hBottom_1", "Waarde"]

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.plot(self.eag.water.level.index, self.eag.water.level, label="berekend peil")
        
        if label_obs:
            if "Peil" in self.eag.series.columns:
                ax.plot(self.eag.series.Peil.index, self.eag.series.Peil, ls="", 
                        marker=".", c="k", label="peil metingen")
        
        ax.axhline(hTarget, linestyle="dashed", lw=1.5, label="hTarget", color="k")
        
        if hTargetMin != 0.:
            ax.axhline(hTargetMin, linestyle="dashed", lw=1.5, label="hTargetMin", color="b")
        if hTargetMax != 0.:
            ax.axhline(hTargetMax, linestyle="dashed", lw=1.5, label="hTargetMax", color="r")

        ax.axhline(hBottom, linestyle="dashdot", lw=1.5, label="hBottom", color="brown")

        ax.set_ylabel("peil (m NAP)")
        ax.legend(loc="best")

        fig.tight_layout()
        
        return ax

    def plot_series(self, series, tmin="2000", tmax="2015", freq="D", mask=None, labelcol="WaardeAlfa"):
        """Method to plot timeseries based on a pandas DataFrame with series names.

        Parameters
        ----------
        series: pandas.DataFrame
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        freq: str

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)

        for id, df in series.groupby(["BakjeID", "ClusterType", "ParamType"]):
            BakjeID, ClusterType, ParamType = id
            if mask is not None:
                if (BakjeID == mask) or (ClusterType == mask) or (ParamType == mask):
                    for j in range(df.shape[0]):
                        series = get_series(ClusterType, ParamType, df.iloc[j:j+1], tmin, tmax, freq)
                        ax.plot(series.index, series, label=df.iloc[j].loc[labelcol])
            else:
                for j in range(df.shape[0]):
                    series = get_series(ClusterType, ParamType, df, tmin, tmax, freq)
                    ax.plot(series.index, series, label=df.iloc[j].loc[labelcol])

        ax.legend(loc="best")
        fig.tight_layout()

        return ax
