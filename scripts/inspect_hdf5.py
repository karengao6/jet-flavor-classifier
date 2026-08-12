from pathlib import Path

import h5py


def print_structure(name: str, obj: h5py.Dataset | h5py.Group) -> None:
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")


def main() -> None:
    path = Path("data/raw/mc-flavtag-ttbar-small.h5")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as f:
        print("Top-level keys:")
        for key in f.keys():
            print(f"  {key}")

        print("\nDatasets:")
        f.visititems(print_structure)


if __name__ == "__main__":
    main()