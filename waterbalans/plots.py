"""The Plot module contains the methods for plotting of a waterbalans.

Author: R.A. Collenteur, Artesia Water, 2017-11-20
        D.A. Brakenhoff, Artesia Water, 2018-09-01
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import colors as mcolors
from pandas import Series


class Eag_Plots:
    def __init__(self, eag, dpi=100):
        self.eag = eag
        self.colordict = OrderedDict(
            {
                "kwel": "brown",
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
                "sluitfout": "black",
            }
        )
        self.figsize = (18, 6)
        self.dpi = dpi

    def series(self):

        fig, axgr = plt.subplots(
            len(self.eag.series.columns),
            1,
            sharex=True,
            figsize=self.figsize,
            dpi=self.dpi,
        )

        for icol, iax in zip(self.eag.series.columns, axgr):
            iax.plot(
                self.eag.series.index, self.eag.series.loc[:, icol], label=icol
            )
            iax.grid(b=True)
            iax.legend(loc="best")

        fig.tight_layout()
        return axgr

    def bucket(self, name, freq="M", tmin=None, tmax=None):
        bucket = self.eag.buckets[name]

        # get tmin, tmax if not defined
        if tmin is None:
            tmin = bucket.fluxes.index[0]
        if tmax is None:
            tmax = bucket.fluxes.index[-1]

        # get data
        plotdata = (
            bucket.fluxes.loc[tmin:tmax].astype(float).resample(freq).mean()
        )

        # get correct colors per flux
        rgbcolors = []
        i = 0
        for icol in plotdata.columns:
            if icol in self.colordict.keys():
                rgbcolors.append(mcolors.to_rgba(self.colordict[icol]))
            else:
                # if no color defined get one of the default ones
                rgbcolors.append(mcolors.to_rgba("C{}".format(i % 10)))
                i += 1

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax = plotdata.plot.bar(stacked=True, width=1, color=rgbcolors, ax=ax)

        ax.set_title("{}: bakje {}".format(self.eag.name, bucket.name))
        ax.set_ylabel("<- uitstroming | instroming ->")
        ax.grid(axis="y")

        ax.legend(ncol=2)

        # set ticks (currently only correct for monthly data)
        if freq == "M":
            ax.set_xticklabels(
                [dt.strftime("%b-%y") for dt in plotdata.index.to_pydatetime()]
            )
        fig.tight_layout()

        return ax

    def aggregated(self, freq="M", tmin=None, tmax=None, add_gemaal=False):

        if add_gemaal:
            fluxes = self.eag.aggregate_fluxes_w_pumpstation()
        else:
            fluxes = self.eag.aggregate_fluxes()

        # get tmin, tmax if not defined
        if tmin is None:
            tmin = fluxes.index[0]
        if tmax is None:
            tmax = fluxes.index[-1]

        # get data and sort columns
        plotdata = fluxes.loc[tmin:tmax].astype(float).resample(freq).mean()
        missing_cols = fluxes.columns.difference(self.colordict.keys()).tolist()
        column_order = [
            k for k in self.colordict.keys() if k in fluxes.columns
        ] + missing_cols
        plotdata = plotdata.loc[:, column_order]

        # get correct colors per flux
        rgbcolors = []
        i = 0
        for icol in plotdata.columns:
            if icol in self.colordict.keys():
                rgbcolors.append(mcolors.to_rgba(self.colordict[icol]))
            else:
                # if no color defined get one of the default ones
                rgbcolors.append(mcolors.to_rgba("C{}".format(i % 10)))
                i += 1

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax = plotdata.plot.bar(stacked=True, width=1, color=rgbcolors, ax=ax)

        if freq == "Y":
            mask = np.ones(len(plotdata.index), dtype="bool")
            fmt = "%Y"
        elif freq == "M":
            fmt = "%Y-%b"
            if plotdata.shape[0] > 24:
                mask = (
                    (plotdata.index.month == 1)
                    | (plotdata.index.month == 4)
                    | (plotdata.index.month == 7)
                    | (plotdata.index.month == 10)
                )
            else:
                mask = np.ones(len(plotdata.index), dtype="bool")
        elif freq == "D":
            fmt = "%Y-%b-%d"
            mask = plotdata.index.day == 1

        ticklabels = plotdata.index.strftime(fmt).to_numpy()
        ticklabels[~mask] = ""
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))

        # fixes the tracker: https://matplotlib.org/users/recipes.html
        def formatter(x, pos=0, max_i=len(ticklabels) - 1):
            i = int(x)
            i = 0 if i < 0 else max_i if i > max_i else i
            return plotdata.index[i].strftime(fmt)

        ax.fmt_xdata = formatter

        # set legends and labels
        ax.set_title(self.eag.name)
        ax.set_ylabel("<- uitstroming | instroming ->")
        ax.grid(axis="y")
        ax.legend(ncol=4)

        # format figure
        ax.figure.autofmt_xdate()
        fig.tight_layout()

        return ax

    def gemaal(self, tmin=None, tmax=None):
        fluxes = self.eag.aggregate_fluxes()

        # get tmin, tmax if not defined
        if tmin is None:
            tmin = fluxes.index[0]
        if tmax is None:
            tmax = fluxes.index[-1]

        fluxes = fluxes.loc[tmin:tmax]
        calculated_out = fluxes.loc[:, ["berekende uitlaat"]]

        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax.plot(
            calculated_out.index,
            -1 * calculated_out,
            lw=2,
            label="berekende uitlaat",
        )

        gemaal_cols = [
            icol
            for icol in self.eag.series.columns
            if icol.lower().startswith("gemaal")
        ]
        if len(gemaal_cols) > 0:
            gemaal = (
                self.eag.series.loc[:, gemaal_cols].loc[tmin:tmax].sum(axis=1)
            )
            ax.plot(gemaal.index, gemaal, lw=2, label="gemeten bij gemaal")

        ax.set_ylabel("Afvoer (m$^3$/dag)")
        ax.legend(loc="best")
        ax.grid(b=True)
        fig.tight_layout()

        return ax

    def cumsum_series(
        self,
        fluxes_names=("berekende inlaat", "berekende uitlaat"),
        eagseries_names=("Gemaal",),
        tmin=None,
        tmax=None,
        period="year",
        month_offset=9,
    ):

        if len(self.eag.series.columns.intersection(set(eagseries_names))) > 0:
            cumsum_fluxes, cumsum_series = self.eag.calculate_cumsum(
                fluxes_names=fluxes_names,
                eagseries_names=eagseries_names,
                cumsum_period=period,
                month_offset=month_offset,
                tmin=tmin,
                tmax=tmax,
            )
        else:
            cumsum_fluxes = self.eag.calculate_cumsum(
                fluxes_names=fluxes_names,
                eagseries_names=None,
                cumsum_period=period,
                month_offset=month_offset,
                tmin=tmin,
                tmax=tmax,
            )
            cumsum_series = None

        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        for flux_nam in fluxes_names:
            ax.plot(
                cumsum_fluxes[flux_nam].index,
                cumsum_fluxes[flux_nam],
                lw=2,
                label=flux_nam,
            )

        if cumsum_series is not None:
            for i, eseries_nam in enumerate(eagseries_names):
                if i == 0:
                    ylower = 0.0
                    yupper = -cumsum_series[eseries_nam]
                else:
                    ylower = -cumsum_series.loc[
                        :, [eagseries_names[j] for j in range(i)]
                    ].sum(axis=1)
                    yupper = -cumsum_series.loc[
                        :, [eagseries_names[j] for j in range(i + 1)]
                    ].sum(axis=1)
                ax.fill_between(
                    cumsum_series[eseries_nam].index,
                    ylower,
                    yupper,
                    label=eseries_nam,
                )

        ax.set_ylabel("Cumulatieve afvoer (m$^3$)")
        ax.legend(loc="best")
        ax.grid(b=True)
        fig.tight_layout()

        return ax

    def wq_concentration(self, c, tmin=None, tmax=None):
        # get tmin, tmax if not defined
        if tmin is None:
            tmin = self.eag.series.index[0]
        if tmax is None:
            tmax = self.eag.series.index[-1]

        c = c.loc[tmin:tmax]

        # Plot
        _, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax.plot(c.index, c, label=self.eag.name)
        ax.grid(b=True)
        ax.legend(loc="best")
        ax.set_ylabel("Concentration (mg/L)")

        return ax

    def fractions(self, tmin=None, tmax=None, concentration=None):
        # get tmin, tmax if not defined
        if tmin is None:
            tmin = self.eag.series.index[0]
        if tmax is None:
            tmax = self.eag.series.index[-1]

        colordict = OrderedDict(
            {
                "kwel": "#e24b00",
                "neerslag": "#0000ff",
                "verhard": "#808080",
                "uitspoeling": "#00ff00",
                "afstroming": "#008000",
                "drain": "#ff9900",
                "inlaat1": "#ff00ff",
                "inlaat2": "#ff99cc",
                "inlaat3": "#953735",
                "inlaat4": "#e24b00",
                "berekende inlaat": "#ff0000",
                "q_cso": "#000000",
            }
        )

        fr = self.eag.calculate_fractions().loc[tmin:tmax]
        fr.dropna(how="all", axis=1, inplace=True)

        # add initial
        fr_list = [fr["initial"].astype(np.float).values]
        labels = ["initieel"]
        colors = ["#c0c0c0"]

        # loop through colordict to determine order
        for icol, c in colordict.items():
            if icol in fr.columns:
                if fr[icol].dropna().shape[0] > 1:
                    fr_list.append(fr[icol].astype(np.float).values)
                    colors.append(c)
                    labels.append(icol)

        # add any missing series
        m = 0
        for icol in fr.columns.difference(colordict.keys()):
            if icol not in [
                "verdamping",
                "wegzijging",
                "berekende uitlaat",
                "initial",
                "intrek",
            ]:
                if fr[icol].astype(np.float).sum() != 0.0:
                    fr_list.append(fr[icol].astype(np.float).values)
                    colors.append(mcolors.to_rgba(f"C{m}"))
                    labels.append(icol)
                    m += 1

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

    def wq_loading(self, mass_in, mass_out, tmin=None, tmax=None, freq="Y"):

        # get tmin, tmax if not defined
        if tmin is None:
            tmin = mass_in.index[0]
        if tmax is None:
            tmax = mass_in.index[-1]

        plotdata_in = (
            mass_in.loc[tmin:tmax].resample(freq).mean()
            / self.eag.water.area
            * 1e3
        )
        plotdata_out = (
            mass_out.loc[tmin:tmax].resample(freq).mean()
            / self.eag.water.area
            * 1e3
        )

        # get correct colors per flux
        rgbcolors_in = []
        i = 0
        for icol in plotdata_in.columns:
            if icol in self.colordict.keys():
                rgbcolors_in.append(mcolors.to_rgba(self.colordict[icol]))
            else:
                # if no color defined get one of the default ones
                rgbcolors_in.append(mcolors.to_rgba("C{}".format(i % 10)))
                i += 1

        rgbcolors_out = []
        i = 0
        for icol in plotdata_out.columns:
            if icol in self.colordict.keys():
                rgbcolors_out.append(mcolors.to_rgba(self.colordict[icol]))
            else:
                # if no color defined get one of the default ones
                rgbcolors_out.append(mcolors.to_rgba("C{}".format(i % 10)))
                i += 1

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=150)
        ax = plotdata_in.plot.bar(
            stacked=True, width=1, ax=ax, edgecolor="k", color=rgbcolors_in
        )

        ax = plotdata_out.plot.bar(
            stacked=True, width=1, ax=ax, edgecolor="k", color=rgbcolors_out
        )
        ax.set_title(self.eag.name)
        ax.set_ylabel("belasting (mg/m$^2$/d)")
        ax.grid(axis="y")

        ax.legend(ncol=4)

        # set ticks
        if freq == "M":
            ax.set_xticks(ax.get_xticks()[::2])
            ax.set_xticklabels(
                [
                    dt.strftime("%b-%y")
                    for dt in plotdata_in.index.to_pydatetime()[::2]
                ]
            )
            # ax.xaxis.set_major_locator(mdates.YearLocator())
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        elif freq == "Y":
            ax.set_xticklabels(
                [dt.strftime("%Y") for dt in plotdata_in.index.to_pydatetime()]
            )

        fig.tight_layout()

        return ax

    def water_level(self, plot_obs=None):

        hTarget = self.eag.water.parameters.loc["hTarget_1", "Waarde"]

        add_target_levels = True
        if self.eag.water.hTargetSeries.empty:
            hTargetMax = self.eag.water.parameters.loc["hTargetMax_1", "Waarde"]
            hTargetMin = self.eag.water.parameters.loc["hTargetMin_1", "Waarde"]
            if hTargetMax == -9999.0 or hTargetMin == -9999.0:
                add_target_levels = False
        else:
            hTargetMax = self.eag.water.hTargetSeries.hTargetMax
            hTargetMin = self.eag.water.hTargetSeries.hTargetMin

        hBottom = self.eag.water.parameters.loc["hBottom_1", "Waarde"]

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        ax.plot(
            self.eag.water.level.index,
            self.eag.water.level,
            label="berekend peil",
        )

        if plot_obs is None or plot_obs:
            if self.eag.water.use_waterlevel_series:
                if "Peil" in self.eag.series.columns:
                    ax.plot(
                        self.eag.series.Peil.index,
                        self.eag.series.Peil,
                        ls="",
                        marker=".",
                        c="k",
                        label="peil metingen",
                    )
        else:
            ax.axhline(
                hTarget, linestyle="dashed", lw=1.5, label="hTarget", color="k"
            )

        if add_target_levels:
            if isinstance(hTargetMin, Series):
                ax.plot(
                    hTargetMin.index,
                    hTargetMin,
                    linestyle="dashed",
                    lw=1.5,
                    label="hTargetMin",
                    color="r",
                )
                ax.plot(
                    hTargetMax.index,
                    hTargetMax,
                    linestyle="dashed",
                    lw=1.5,
                    label="hTargetMax",
                    color="b",
                )
            else:
                if hTargetMin < 0:
                    ax.axhline(
                        hTarget - np.abs(hTargetMin),
                        linestyle="dashed",
                        linewidth=1.5,
                        label="hTargetMin",
                        color="r",
                    )
                else:
                    ax.plot(
                        self.eag.series.Peil.index,
                        self.eag.series.Peil - np.abs(hTargetMin),
                        c="r",
                        ls="dashed",
                        lw=1.5,
                        label="hTargetMin",
                    )
                if hTargetMax < 0:
                    ax.axhline(
                        hTarget + np.abs(hTargetMax),
                        linestyle="dashed",
                        linewidth=1.5,
                        label="hTargetMax",
                        color="b",
                    )
                else:
                    ax.plot(
                        self.eag.series.Peil.index,
                        self.eag.series.Peil + np.abs(hTargetMax),
                        c="b",
                        ls="dashed",
                        lw=1.5,
                        label="hTargetMax",
                    )

        ax.axhline(
            hBottom, linestyle="dashdot", lw=1.5, label="hBottom", color="C2"
        )

        ax.set_ylabel("peil (m NAP)")
        ax.legend(loc="best")

        fig.tight_layout()

        return ax

    def compare_fluxes_to_excel_balance(
        self, exceldf, showdiff=True
    ):  # pragma: no cover
        """Convenience method to compare original Excel waterbalance to the one
        calculated with Python.

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
        column_names = {
            "peil": 0,
            "neerslag": 1,
            "kwel": 2,
            "verhard": 3,
            "q_cso": 4,
            "drain": 5,
            "uitspoeling": 6,
            "afstroming": 7,
            "inlaat1": 8,
            "inlaat2": 9,
            "inlaat3": 10,
            "inlaat4": 11,
            "berekende inlaat": 12,
            "verdamping": 13,
            "wegzijging": 14,
            "intrek": 15,
            "uitlaat1": 16,
            "uitlaat2": 17,
            "uitlaat3": 18,
            "uitlaat4": 19,
            "berekende uitlaat": 20,
        }

        fluxes = self.eag.aggregate_fluxes()
        fluxes.dropna(how="all", axis=1, inplace=True)
        # drop last day which isn't simulated (day after tmax)
        fluxes = fluxes.iloc[:-1]

        # Plot
        fig, axgr = plt.subplots(
            int(np.ceil((fluxes.shape[1] + 1) / 3)),
            3,
            figsize=(20, 12),
            dpi=self.dpi,
            sharex=True,
        )

        for i, pycol in enumerate(fluxes.columns):
            iax = axgr.ravel()[i]

            iax.plot(
                fluxes.index,
                fluxes.loc[:, pycol],
                label="{} (Python)".format(pycol),
            )
            # hacky method to subtract excel series from diff
            diff = fluxes.loc[:, pycol].copy()

            if (
                pycol not in exceldf.columns
                and pycol not in column_names.keys()
            ):
                self.eag.logger.warning(
                    "Column '{}' not found in Excel Balance!".format(pycol)
                )
                iax.legend(loc="best")
                iax.grid(b=True)
                continue
            else:
                try:
                    excol = column_names[pycol]
                except KeyError:
                    excol = pycol

            iax.plot(
                exceldf.index,
                exceldf.iloc[:, excol],
                label="{0:s} (Excel)".format(
                    exceldf.columns[excol].split(".")[0]
                ),
                ls="dashed",
            )

            iax.grid(b=True)
            iax.legend(loc="best")

            if showdiff:
                iax2 = iax.twinx()
                # hacky method to subtract excel balance (diff column names)
                diff -= exceldf.iloc[:, excol]
                iax2.plot(diff.index, diff, c="C4", lw=0.75)
                yl = np.max(np.abs(iax2.get_ylim()))
                iax2.set_ylim(-1 * yl, yl)

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

        iax = axgr.ravel()[i + 1]
        iax.plot(
            self.eag.water.level.iloc[1:].index,
            self.eag.water.level.iloc[1:],
            label="Berekend peil (Python)",
        )
        iax.plot(
            exceldf.index,
            exceldf.loc[:, "peil"],
            label="Berekend peil (Excel)",
            ls="dashed",
        )
        iax.grid(b=True)
        iax.legend(loc="best")

        if showdiff:
            iax2 = iax.twinx()
            diff = self.eag.water.level.level - exceldf.loc[:, "peil"]
            iax2.plot(diff.index, diff, c="C4", lw=0.75)
            yl = np.max(np.abs(iax2.get_ylim()))
            iax2.set_ylim(-1 * yl, yl)

            mint = np.max([self.eag.water.level.index[0], exceldf.index[0]])
            maxt = np.min([self.eag.water.level.index[-1], exceldf.index[-1]])

            # add check if series are similar (1cm absolute + 1% error on top of value in excel)
            check = np.allclose(
                self.eag.water.level.loc[mint:maxt, "level"],
                exceldf.loc[mint:maxt, "peil"],
                atol=0.005,
                rtol=0.00,
            )

            if check:
                iax.patch.set_facecolor("lightgreen")
                iax.patch.set_alpha(0.5)
            else:
                iax.patch.set_facecolor("salmon")
                iax.patch.set_alpha(0.5)

        fig.tight_layout()
        return fig

    def compare_waterlevel_to_excel(self, exceldf):  # pragma: no cover
        """Convenience method to compare calculated water level in Excel
        waterbalance to the one calculated with Python.

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

        ax.plot(
            self.eag.water.level.index[1:],
            self.eag.water.level.iloc[1:],
            label="Berekend peil (Python)",
        )
        ax.plot(
            exceldf.index,
            exceldf.loc[:, "peil"],
            label="Berekend peil (Excel)",
            ls="dashed",
        )
        ax.grid(b=True)
        ax.legend(loc="best")
        fig.tight_layout()
        return ax
