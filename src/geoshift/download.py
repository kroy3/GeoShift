"""Download and pre-process the public datasets.

QM9 and (r)MD17 are fetched automatically through PyTorch Geometric. ANI-1x
must be downloaded manually because of its size (see README, "Datasets").

Examples::

    geoshift-download                                   # everything used by configs/
    geoshift-download --datasets qm9 md17:aspirin
"""

from __future__ import annotations

import argparse

from geoshift.data import ANI1X_FILENAME, ANI1X_URL, load_source

DEFAULT_DATASETS = [
    "qm9",
    "md17:aspirin",
    "md17:benzene",
    "md17:ethanol",
    "md17:malonaldehyde",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset specs to fetch.")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args(argv)

    for spec in args.datasets:
        if spec.startswith("ani1x"):
            continue
        print(f"Preparing {spec} ...")
        samples = load_source(spec, args.data_dir)
        print(f"  {len(samples)} molecules")

    print(
        f"\nANI-1x: download '{ANI1X_FILENAME}' from\n  {ANI1X_URL}\n"
        f"and place it at {args.data_dir}/ani1x/{ANI1X_FILENAME}"
    )


if __name__ == "__main__":
    main()
