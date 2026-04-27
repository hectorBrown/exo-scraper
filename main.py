from astroquery.mast import Observations

def main():
    obs_table = Observations.query_object("M8", radius=".02 deg")
    df = obs_table.to_pandas()

    print("Column names: \n", list(df))
    print()
    print(df.describe())



if __name__ == "__main__":
    main()
