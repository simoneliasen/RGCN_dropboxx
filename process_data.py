import pathlib as pl

import pandas as pd


def main():
    data_dir = pl.Path("data")

    for dataset in ["wn18rr"]:
        data = pd.concat(
            [
                pd.read_csv(
                    data_dir / dataset / f"{split}.txt",
                    sep="\t",
                    names=["head", "relation", "tail"],
                    dtype=str,
                )
                for split in ["train", "valid", "test"]
            ],
            ignore_index=True,
        )

        entities = pd.concat([data["head"], data["tail"]], ignore_index=True).unique()
        relations = data["relation"].unique()

        with open(data_dir / dataset / "entities.dict", "w") as file:
            file.write(
                "\n".join([f"{n}\t{entity}" for n, entity in enumerate(entities)])
            )

        with open(data_dir / dataset / "relations.dict", "w") as file:
            file.write(
                "\n".join([f"{n}\t{relation}" for n, relation in enumerate(relations)])
            )


if __name__ == "__main__":
    main()
