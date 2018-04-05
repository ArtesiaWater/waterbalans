def test_functie():
    """Deze functie neemt geen argumenten, en returned een Pandas.DataFrame.

    Returns
    -------
    data: pandas.DataFrame

    """
    import pandas as pd
    data = pd.read_csv("data_voor_moria.csv")
    return data
