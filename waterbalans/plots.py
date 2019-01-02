"""The Plot module contains the methods for plotting of a waterbalans.

Author: R.A. Collenteur, Artesia Water, 2017-11-20

"""
import matplotlib.pyplot as plt
from matplotlib import colors


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

    def gemaal_cumsum(self, tmin=None, tmax=None, year_reset=True):
        fluxes = self.eag.aggregate_fluxes()
        
        # get tmin, tmax if not defined
        if tmin is None:
            fluxes.index[0]
        if tmax is None:
            fluxes.index[-1]
        
        fluxes = fluxes.loc[tmin:tmax]
        calculated_out = fluxes.loc[:, ["berekende uitlaat"]]

        if year_reset:
            calculated_out = calculated_out.groupby(by=calculated_out.index.year).cumsum()
        else:
            calculated_out = calculated_out.cumsum()
        
        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.plot(calculated_out.index, -1*calculated_out, lw=2, label="berekende uitlaat")
        
        if "Gemaal" in self.eag.series.columns:
            gemaal = self.eag.series["Gemaal"].loc[tmin:tmax]
            if year_reset:
                gemaal = gemaal.groupby(by=gemaal.index.year).cumsum()
            else:
                gemaal = gemaal.cumsum()    
            ax.fill_between(gemaal.index, 0.0, gemaal, label="gemeten bij gemaal", color="C1")

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
    
    def chloride_fractions(self, tmin=None, tmax=None):
        # get tmin, tmax if not defined
        if tmin is None:
            self.eag.series.index[0]
        if tmax is None:
            self.eag.series.index[-1]

        colordict = {"initieel": "gray",
                     "kwel": "brown", 
                     "neerslag": "blue", 
                     "verhard": "darkgray", 
                     "uitspoeling": "lime", 
                     "afstroming": "darkgreen", 
                     "drain": "orange", 
                     "inlaat": "red", 
                     "q_cso": "black" }
        
        fr = self.eag.calculate_fractions().loc[tmin:tmax]
        init = 1.0 - fr.cumsum().sum(axis=1)

        fr_list = [init.cumsum().values]
        labels = ["initieel"]
        colors = ["gray"]
        for icol, c in colordict.items():
            if icol in fr.columns:
                fr_list.append(fr[icol].cumsum().values)
                colors.append(c)
                labels.append(icol)
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
        ax.stackplot(fr.index, *fr_list, labels=labels, colors=colors)
        ax.grid(b=True)
        ax.legend(loc="best")
        ax.set_ylabel("Percentage (%)")

        return ax
