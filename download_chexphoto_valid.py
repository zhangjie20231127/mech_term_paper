from __future__ import annotations

from pathlib import Path

import redivis


OUTPUT_DIR = Path("/home/zhang/dataset/chexphoto")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    user = redivis.user("aimi")
    dataset = user.dataset("chexphoto:2qwg:v1_0")

    # Label table exposed on the dataset page.
    valid_csv_table = dataset.table("valid.csv")
    valid_csv_df = valid_csv_table.to_pandas_dataframe()
    valid_csv_path = OUTPUT_DIR / "valid.csv"
    valid_csv_df.to_csv(valid_csv_path, index=False)
    print(f"saved labels to {valid_csv_path} rows={len(valid_csv_df)}")

    # File-index table; each row references one image file.
    valid_files_table = dataset.table("valid:hzw4")
    valid_files_df = valid_files_table.to_pandas_dataframe()
    valid_index_path = OUTPUT_DIR / "valid_index.csv"
    valid_files_df.to_csv(valid_index_path, index=False)
    print(f"saved file index to {valid_index_path} rows={len(valid_files_df)}")

    valid_files_dir = OUTPUT_DIR / "valid"
    valid_files_dir.mkdir(parents=True, exist_ok=True)
    valid_files_table.download_files(valid_files_dir)
    print(f"downloaded files into {valid_files_dir}")


if __name__ == "__main__":
    main()
