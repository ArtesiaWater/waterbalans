from .test_run_eag import test_eag_run, test_calculate_chloride
import matplotlib.pyplot as plt

def test_plot_aggregated():
    e = test_eag_run()
    ax = e.plot.aggregated()
    plt.close(ax.figure)
    return

def test_plot_bucket():
    e = test_eag_run()
    ax = e.plot.bucket(name=139640116)
    plt.close(ax.figure)
    return

def test_plot_chloride():
    e, cl = test_calculate_chloride()
    ax = e.plot.chloride(cl)
    plt.close(ax.figure)
    return

def test_plot_chloride_fractions():
    e = test_eag_run()
    ax = e.plot.chloride_fractions(tmin="2000", tmax="2000-01-10")
    plt.close(ax.figure)
    return

def test_plot_waterlevel():
    e = test_eag_run()
    ax = e.plot.water_level()
    plt.close(ax.figure)
    return
