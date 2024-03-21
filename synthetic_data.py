"""
generates synthetic customer data for the notebook illustration of pandas / polars performance
"""
import gzip
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from faker import Faker
from tqdm import tqdm


def generate_data_chunk(seed, num_records_chunk):
    """
    Generate a chunk of synthetic data.
    """
    np.random.seed(seed)
    Faker.seed(seed)
    fake = Faker()

    customer_ids = [fake.passport_number() for _ in range(num_records_chunk)]
    ages = np.random.normal(40, 15, num_records_chunk).astype(int)
    lucky_numbers = np.random.poisson(lam=1.0, size=num_records_chunk).astype(int)

    def random_choices(choices, num_records):
        return np.random.choice(choices, num_records)

    occupations = random_choices(
        [
            "Python Dev",
            "Data Engineer",
            "Data Scientist",
            "Machine Learning Eng",
            "DevOps Savior",
            "Pandas Guru",
            "Polars Guru",
            "Apache Arrow Understudy",
            "Rustacean",
        ],
        num_records_chunk,
    )
    membership_statuses = random_choices(
        ["Supporting", "Managing", "Contributing", "Fellow", "Not Yet a Member"],
        num_records_chunk,
    )
    educations = random_choices(
        ["High School", "College", "Graduate", "Leetcode + sweat", "Raised by Wolves"],
        num_records_chunk,
    )
    date_started_python = pd.to_datetime(
        [fake.date_of_birth(minimum_age=12, maximum_age=100) for _ in range(num_records_chunk)]
    )

    df_chunk = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "their_lucky_number": lucky_numbers,
            "occupation": occupations,
            "psf_membership_status": membership_statuses,
            "education": educations,
            "date_started_python": date_started_python,
        }
    )

    return df_chunk


def update_progress(result):
    """
    Callback function to update progress.  ~5x faster than fancy tdqm progress bar
    """
    global progress
    progress += 1
    print(f"Progress: {progress / num_chunks * 100:.0f}%")


def generate_synthetic_data(num_records: int, seed: int = 77) -> pd.DataFrame:
    """
    Generate synthetic data for testing using multiprocessing.
    """
    num_cpus = psutil.cpu_count(logical=False)
    pool = Pool(processes=num_cpus)

    # to have more even progress reporting, define a fixed chunk size and calculate the number of chunks
    fixed_chunk_size = 100000  # adjust this value based on total # records
    global num_chunks
    num_chunks = (num_records + fixed_chunk_size - 1) // fixed_chunk_size

    global progress
    progress = 0

    results = []
    for i in range(num_chunks):
        start_index = i * fixed_chunk_size
        end_index = min(start_index + fixed_chunk_size, num_records)
        seed_offset = i
        num_records_chunk = end_index - start_index
        result = pool.apply_async(generate_data_chunk, args=(seed + seed_offset, num_records_chunk), callback=update_progress)
        results.append(result)

    # Combine the chunks into a single DataFrame
    df = pd.concat([result.get() for result in results], ignore_index=True)

    return df


def gzip_csv(file_path: str) -> None:
    """
    Compresses a CSV file using gzip and saves it in the same directory.
    """
    source_path = Path(file_path)
    compressed_file_path = source_path.with_suffix(".csv.gz")

    if not compressed_file_path.is_file():
        with source_path.open("rb") as f_in, gzip.open(
                compressed_file_path, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Compressed file saved as {compressed_file_path}")
    else:
        print("Zipped copy of datafile already exists")


def generate_and_save_synthetic_data(num_records: int = 1000):
    data_dir = Path("data")
    csv_filename = data_dir / "python_dev_universe.csv"
    gzipped_csv_filename = data_dir / "python_dev_universe.csv.gz"
    parquet_filename = data_dir / "python_dev_universe.parquet"

    data_dir.mkdir(parents=True, exist_ok=True)

    if (
            not csv_filename.is_file()
            and not gzipped_csv_filename.is_file()
            and not parquet_filename.is_file()
    ):
        print("test data missing - generating millions of records")
        print(f"saving all data to {data_dir} directory")
        df = generate_synthetic_data(num_records=num_records)

        print("saving to csv")
        df.to_csv(csv_filename, index=False)

        print("saving gzipped csv")
        gzip_csv(csv_filename)

        print("converting to parquet and saving local parquet file")
        df.to_parquet(parquet_filename, index=False)

    else:
        print("Test data already exists in multiple formats.")
        # Add file size info here


if __name__ == "__main__":
    generate_and_save_synthetic_data(num_records=10000000)
