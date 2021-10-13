import matplotlib.pyplot as plt
from .test_002_run_eag import test_eag_run, test_calculate_chloride


def test_plot_aggregated():
    e = test_eag_run()
    ax = e.plot.aggregated()
    plt.close(ax.figure)
    return


def test_plot_bucket():
    e = test_eag_run()
    ax = e.plot.bucket(name=1396122676)
    plt.close(ax.figure)
    return


def test_plot_chloride():
    e, m = test_calculate_chloride()
    ax = e.plot.wq_concentration(m[-1] / e.water.storage["storage"])
    plt.close(ax.figure)
    return


def test_plot_chloride_fractions():
    e = test_eag_run()
    ax = e.plot.fractions(tmin="2000", tmax="2000-01-10")
    plt.close(ax.figure)
    return


def test_plot_waterlevel():
    e = test_eag_run()
    ax = e.plot.water_level()
    plt.close(ax.figure)
    return
