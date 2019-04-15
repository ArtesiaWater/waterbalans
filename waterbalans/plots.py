"""The Plot module contains the methods for plotting of a waterbalans.

Author: R.A. Collenteur, Artesia Water, 2017-11-20
        D.A. Brakenhoff, Artesia Water, 2018-09-01

"""
from collections import OrderedDict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from pandas import Timedelta, Timestamp

from .timeseries import get_series


class Eag_Plots:
    def __init__(self, eag, dpi=150):
        self.eag = eag
        self.colordict =  OrderedDict(
                          {"kwel": "brown",
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
                           "berekende uitlaat": "k",
                           "maalstaat": "yellow",
                           "sluitfout": "black"})
        self.figsize = (18, 6)
        self.dpi = dpi

    def series(self):

        fig, axgr = plt.subplots(len(self.eag.series.columns), 1, sharex=True)

        for icol, iax in zip(self.eag.series.columns, axgr):
            iax.plot(self.eag.series.index, self.eag.series.loc[:, icol], label=icol)
            iax.grid(b=True)
            iax.legend(loc="best")
        
        fig.tight_layout()
        return axgr

    def plot_series(self, series, tmin="2000", tmax="2018", freq="D", mask=None, labelcol="WaardeAlfa"):  # pragma: no cover
        """Method to plot timeseries based on a pandas DataFrame with series names.

        Parameters
        ----------
        series: pandas.DataFrame
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        freq: str

        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)

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
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
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
    
    def aggregated(self, freq="M", tmin=None, tmax=None, add_gemaal=False):
        if add_gemaal:
            fluxes = self.eag.aggregate_with_pumpstation()
        else:
            fluxes = self.eag.aggregate_fluxes()

        # get tmin, tmax if not defined
        if tmin is None:
            fluxes.index[0]
        if tmax is None:
            fluxes.index[-1]

        # get data and sort columns
        plotdata = fluxes.loc[tmin:tmax].astype(float).resample(freq).mean()
        missing_cols = fluxes.columns.difference(self.colordict.keys()).tolist()
        column_order = [k for k in self.colordict.keys() if k in fluxes.columns] + missing_cols
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
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax = plotdata.plot.bar(stacked=True, width=1, color=rgbcolors, ax=ax)
        
        ax.set_title(self.eag.name)
        ax.set_ylabel("<- uitstroming | instroming ->")
        ax.grid(axis="y")
        
        ax.legend(ncol=4)

        # set ticks (currently only correct for monthly data)
        if freq == "M":
            ax.set_xticks(ax.get_xticks()[::2])
            ax.set_xticklabels([dt.strftime('%b-%y') for dt in 
                                plotdata.index.to_pydatetime()[::2]])
            
            # ax.xaxis.set_major_locator(mdates.YearLocator())
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

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
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax.plot(calculated_out.index, -1*calculated_out, lw=2, label="berekende uitlaat")
        
        if "Gemaal" in self.eag.series.columns:
            gemaal = self.eag.series["Gemaal"].loc[tmin:tmax]
            ax.plot(gemaal.index, gemaal, lw=2, label="gemeten bij gemaal")

        ax.set_ylabel("Afvoer (m$^3$/dag)")
        ax.legend(loc="best")
        ax.grid(b=True)
        fig.tight_layout()

        return ax

    def cumsum_series(self, fluxes_names=["berekende inlaat", "berekende uitlaat"], 
                      eagseries_names=["Gemaal"], tmin=None, tmax=None, 
                      period="year", month_offset=9):
        
        if len(self.eag.series.columns.intersect(set(eagseries_names))) > 0:
            cumsum_fluxes, cumsum_series = self.eag.cumulative_period_sum(fluxes_names=fluxes_names,
                                                                          eagseries_names=eagseries_names,
                                                                          cumsum_period=period, 
                                                                          month_offset=month_offset,
                                                                          tmin=tmin, tmax=tmax)
        else:
            cumsum_fluxes = self.eag.cumulative_period_sum(fluxes_names=fluxes_names,
                                                           eagseries_names=None,
                                                           cumsum_period=period, 
                                                           month_offset=month_offset,
                                                           tmin=tmin, tmax=tmax)
            cumsum_series = None

        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        for flux_nam in fluxes_names:
            ax.plot(cumsum_fluxes[flux_nam].index, cumsum_fluxes[flux_nam], lw=2, label=flux_nam)

        if cumsum_series is not None:
            for eseries_nam in eagseries_names:
                ax.fill_between(cumsum_series[eseries_nam].index, 0.0, -cumsum_series[eseries_nam], 
                                label=eseries_nam, color="C1")

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
        _, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax.plot(c.index, c, label=self.eag.name)
        ax.grid(b=True)
        ax.legend(loc="best")
        ax.set_ylabel("Chloride concentratie (mg/L)")

        return ax
    
    def fractions(self, tmin=None, tmax=None, concentration=None):
        # get tmin, tmax if not defined
        if tmin is None:
            self.eag.series.index[0]
        if tmax is None:
            self.eag.series.index[-1]

        colordict = OrderedDict(
                    {"kwel": "brown", 
                     "neerslag": "blue", 
                     "uitspoeling": "lime", 
                     "afstroming": "darkgreen", 
                     "drain": "orange", 
                     "berekende inlaat": "red", 
                     "q_cso": "black",
                     "verhard": "gray"})
        
        fr = self.eag.calculate_fractions().loc[tmin:tmax]
        fr.dropna(how="all", axis=1, inplace=True)

        fr_list = [fr["initial"].astype(np.float).values]
        labels = ["initieel"]
        colors = ["lightgray"]
        
        # loop through colordict to determine order
        for icol, c in colordict.items():
            if icol in fr.columns:
                if fr[icol].dropna().shape[0] > 1:
                    fr_list.append(fr[icol].astype(np.float).values)
                    colors.append(c)
                    labels.append(icol)
        
        for icol in fr.columns.difference(colordict.keys()):
            if icol not in ["verdamping", "wegzijging", "berekende uitlaat",
                            "initial", "intrek"]:
                if fr[icol].astype(np.float).sum() != 0.0:
                    fr_list.append(fr[icol].astype(np.float).values)
                    colors.append("salmon")
                    labels.append(icol)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax.stackplot(fr.index, *fr_list, labels=labels, colors=colors)
        ax.grid(b=True)
        ax.legend(loc="upper center", ncol=2)
        ax.set_ylabel("Percentage (%)")

        if concentration is not None:
            ax2 = ax.twinx()
            ax2.plot(concentration.index, concentration, color="k")
            ax2.set_ylabel("Concentration (mg/L)")
        
        # ax.set_xlim(Timestamp(tmin), Timestamp(tmax))
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return ax

    def water_level(self, label_obs=False):

        hTarget = self.eag.water.parameters.loc["hTarget_1", "Waarde"]
        # hTargetMax = hTarget + np.abs(self.eag.water.parameters.loc["hTargetMax_1", "Waarde"])
        # hTargetMin = hTarget - np.abs(self.eag.water.parameters.loc["hTargetMin_1", "Waarde"])
        hTargetMax = self.eag.water.hTargetSeries.hTargetMax
        hTargetMin = self.eag.water.hTargetSeries.hTargetMin
        hBottom = self.eag.water.parameters.loc["hBottom_1", "Waarde"]

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        ax.plot(self.eag.water.level.index, self.eag.water.level, label="berekend peil")
        
        if label_obs:
            if "Peil" in self.eag.series.columns:
                ax.plot(self.eag.series.Peil.index, self.eag.series.Peil, ls="", 
                        marker=".", c="k", label="peil metingen")
        
        ax.axhline(hTarget, linestyle="dashed", lw=1.5, label="hTarget", color="k")
        
        ax.plot(hTargetMin.index, hTargetMin, linestyle="dashed", lw=1.5, label="hTargetMin", color="r")
        ax.plot(hTargetMax.index, hTargetMax, linestyle="dashed", lw=1.5, label="hTargetMax", color="b")

        ax.axhline(hBottom, linestyle="dashdot", lw=1.5, label="hBottom", color="brown")

        ax.set_ylabel("peil (m NAP)")
        ax.legend(loc="best")

        fig.tight_layout()
        
        return ax

    def compare_fluxes_to_excel_balance(self, exceldf, showdiff=True):  # pragma: no cover
        """Convenience method to compare original Excel waterbalance
        to the one calculated with Python.
        
        Parameters
        ----------
        exceldf : pandas.DataFrame
            A pandas DataFrame containing the water balance series from 
            the Excel File. Columns "A,AJ,CH:DB" from the "REKENHART" sheet.
        showdiff : bool, optional
            if True show difference between Python and Excel on secondary axes.
        
        Returns
        -------
        fig: matplotlib figure handle
            handle to figure containing N subplots comparing series from 
            Python waterbalance to the Excel waterbalance.

        """
        column_names = {'peil':        0,
                        'neerslag':    1,
                        'kwel':        2,
                        'verhard':     3,
                        'q_cso':       4,
                        'drain':       5,
                        'uitspoeling': 6,
                        'afstroming':  7,
                        'Inlaat1':     8,
                        'Inlaat2':     9,
                        'Inlaat3':    10,
                        'Inlaat4':    11,
                        'berekende inlaat':     12,
                        'verdamping': 13,
                        'wegzijging': 14,
                        'intrek':     15,
                        'Uitlaat1':   16,
                        'Uitlaat2':   17,
                        'Uitlaat3':   18,
                        'Uitlaat4':   19,
                        'berekende uitlaat':    20}

        fluxes = self.eag.aggregate_fluxes()
        fluxes.dropna(how="all", axis=1, inplace=True)
        fluxes = fluxes.iloc[:-1]  # drop last day which isn't simulated (day after tmax)

        # Plot
        fig, axgr = plt.subplots(int(np.ceil((fluxes.shape[1]+1)/3)), 3, 
                                 figsize=(20, 12), dpi=self.dpi, sharex=True)

        for i, pycol in enumerate(fluxes.columns):
            iax = axgr.ravel()[i]

            iax.plot(fluxes.index, fluxes.loc[:, pycol], label="{} (Python)".format(pycol))
            diff = fluxes.loc[:, pycol].copy() # hacky method to subtract excel series from diff

            if pycol not in exceldf.columns and pycol not in column_names.keys():
                print("Column '{}' not found in Excel Balance!".format(pycol))
                iax.legend(loc="best")
                iax.grid(b=True)
                continue
            else:
                try:
                    excol = column_names[pycol]
                except KeyError:
                    excol = pycol

            iax.plot(exceldf.index, exceldf.iloc[:, excol], label="{0:s} (Excel)".format(exceldf.columns[excol].split(".")[0]), 
                     ls="dashed")

            iax.grid(b=True)
            iax.legend(loc="best")

            if showdiff:
                iax2 = iax.twinx()
                diff -= exceldf.iloc[:, excol]  # hacky method to subtract excel balance (diff column names)
                iax2.plot(diff.index, diff, c="C4", lw=0.75)
                yl = np.max(np.abs(iax2.get_ylim()))
                iax2.set_ylim(-1*yl, yl)
                
                # add check if difference is larger than 5% on average
                perc_err = diff / exceldf.iloc[:, excol]
                perc_err.loc[~np.isfinite(perc_err)] = 0.0
                check = perc_err.abs().mean() > 0.05

                if check > 0:
                    iax.patch.set_facecolor("salmon")
                    iax.patch.set_alpha(0.5)
                else:
                    iax.patch.set_facecolor("lightgreen")
                    iax.patch.set_alpha(0.5)
        
        iax = axgr.ravel()[i+1]
        iax.plot(self.eag.water.level.iloc[1:].index, self.eag.water.level.iloc[1:], 
                label="Berekend peil (Python)")
        iax.plot(exceldf.index, exceldf.loc[:, "peil"], 
                label="Berekend peil (Excel)", ls="dashed")
        iax.grid(b=True)
        iax.legend(loc="best")

        if showdiff:
            iax2 = iax.twinx()
            diff = self.eag.water.level.level - exceldf.loc[:, "peil"]
            iax2.plot(diff.index, diff, c="C4", lw=0.75)
            yl = np.max(np.abs(iax2.get_ylim()))
            iax2.set_ylim(-1*yl, yl)
            
            mint = np.max([self.eag.water.level.index[0], exceldf.index[0]])
            maxt = np.min([self.eag.water.level.index[-1], exceldf.index[-1]])

            # add check if series are similar (1cm absolute + 1% error on top of value in excel)
            check = np.allclose(self.eag.water.level.loc[mint:maxt, "level"], 
                                exceldf.loc[mint:maxt, "peil"], 
                                atol=0.005, rtol=0.00)

            if check:
                iax.patch.set_facecolor("lightgreen")
                iax.patch.set_alpha(0.5)
            else:
                iax.patch.set_facecolor("salmon")
                iax.patch.set_alpha(0.5)

        fig.tight_layout()
        return fig

    def compare_waterlevel_to_excel(self, exceldf):  # pragma: no cover
        """Convenience method to compare calculated water level in Excel waterbalance
        to the one calculated with Python.
        
        Parameters
        ----------
        exceldf : pandas.DataFrame
            A pandas DataFrame containing the water balance series from 
            the Excel File. Columns "A,AJ,CH:DB" from the "REKENHART" sheet.
        
        Returns
        -------
        ax: 
            handle to axes containing N subplots comparing series from 
            Python waterbalance to the Excel waterbalance.
        
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=125)

        ax.plot(self.eag.water.level.index[1:], self.eag.water.level.iloc[1:], 
                label="Berekend peil (Python)")
        ax.plot(exceldf.index, exceldf.loc[:, "peil"], 
                label="Berekend peil (Excel)", ls="dashed")
        ax.grid(b=True)
        ax.legend(loc="best")
        fig.tight_layout()
        return ax
