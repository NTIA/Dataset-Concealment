import os
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def write_group(group, output_path):
    with open(output_path, "w") as f:
        yaml.dump(group, f)


def make_concealed_groups(dataset_dict, output_dir):
    concealed_dir = os.path.join(output_dir, "Concealed")
    os.makedirs(concealed_dir, exist_ok=True)
    for name, ds in dataset_dict.items():
        concealed_group = [v for k, v in dataset_dict.items() if v != ds]
        output_path = os.path.join(concealed_dir, f"Conceal-{name}.yaml")
        write_group(concealed_group, output_path)


def make_global_group(dataset_dict, output_dir):
    global_group = [v for k, v in dataset_dict.items()]
    output_path = os.path.join(output_dir, "Global.yaml")
    write_group(global_group, output_path)


def make_individual_groups(dataset_dict, output_dir):
    individual_dir = os.path.join(output_dir, "Individual")
    os.makedirs(individual_dir, exist_ok=True)
    for name, ds in dataset_dict.items():
        individual_group = [ds]
        output_path = os.path.join(individual_dir, f"{name}.yaml")
        write_group(individual_group, output_path)


def main(dataset_dict, output_dir):
    make_concealed_groups(dataset_dict, output_dir)
    make_global_group(dataset_dict, output_dir)
    make_individual_groups(dataset_dict, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_file",
        help=(
            "YAML dictionary with dataset names and paths to folder containing splits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data-groups",
        help="Directory to store dataset group files.",
    )

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        dataset_dict = yaml.safe_load(f)

    main(dataset_dict=dataset_dict, output_dir=args.output_dir)
