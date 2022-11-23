import argparse
import itertools as it
import pathlib as pl
import re


def replace_settings(text, values):
    for key, value in values.items():
        text = replace_setting(text, key, value)

    return text


def replace_setting(text, key, value):
    return re.sub(fr"(.*{key})=.*", fr"\1={value}", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wn18rr", "fb15k237"])
    parser.add_argument("output")
    parser.add_argument("--dimensions", type=int, nargs="*", default=[50])
    parser.add_argument("--batch-size", type=int, nargs="*", default=[64])
    parser.add_argument("--learning-rate", type=float, nargs="*", default=[0.1])
    parser.add_argument("--dropout", type=float, nargs="*", default=[0])
    parser.add_argument("--layers", type=int, nargs="*", default=[1])
    parser.add_argument("--regularisation", type=float, nargs="*", default=[0.01])
    args = parser.parse_args()

    filtered_args = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in ["output", "dataset"]
    }

    keys, values = zip(*sorted(filtered_args.items()))

    output = pl.Path(args.output)
    output.mkdir(exist_ok=True)

    for value in it.product(*values):
        config = dict(zip(keys, value))

        config_key = "-".join(
            ["{}-{}".format(key, value) for key, value in config.items()]
        )
        config_file = output / f"{config_key}.exp"
        script_file = output / f"{config_key}.sh"
        output_file = output / f"{config_key}.out"

        with open("settings/gcn_block.exp", "r") as file:
            config_content = file.read()

        config_content = replace_settings(
            config_content,
            {
                "Dimension": config["dimensions"],
                "ExperimentName": output / config_key,
                "NegativeSampleRate": 1,
                "GraphBatchSize": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "DropoutKeepProbability": 1 - config["dropout"],
                "NumberOfLayers": config["layers"],
                "RegularizationParameter": config["regularisation"],
            },
        )

        with open(config_file, "w") as file:
            file.write(config_content)

        with open(script_file, "w") as file:
            file.write(
                f"(time PYTHONPATH=code/optimization python code/train.py --dataset data/{args.dataset} --settings {config_file}) &> {output_file}"
            )


if __name__ == "__main__":
    main()
