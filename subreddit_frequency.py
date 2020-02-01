import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import argparse
import time
from utils import load_dataframe_from_jsonl


if __name__ == '__main__':
    test_file = "RC_2019-09_ukpolitics.dump"

    start_time = time.time()
    df = load_dataframe_from_jsonl(test_file)
    print(df.head())
    end_time = time.time()
    print(end_time - start_time)
