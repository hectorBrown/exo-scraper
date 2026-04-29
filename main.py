import lightkurve as lk
import sys


def main():
    lc = lk.search_lightcurve("HD39091").download()  # Pi Men
    if lc is None:
        sys.exit(1)

    lc = lc.normalize().flatten()
    lc.interact_bls()


if __name__ == "__main__":
    main()
