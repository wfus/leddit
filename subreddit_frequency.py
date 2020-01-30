import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import argparse
import time


def load_dataframe_from_jsonl(fname):
    """Loads a reddit dump of a non-delimited jsonlist into a pandas dataframe.
    This won't work for files larger than a few gigabytes, the input needs to
    be filtered from the monthly dumps first."""
    dump_file = Path(fname)

    comments = []
    with dump_file.open('r') as f:
        for line in tqdm(f):
            comment = json.loads(line.strip())
            comments.append(comment)

    df = pd.DataFrame(comments)
    return df


if __name__ == '__main__':
    test_file = "RC_2019-09_ukpolitics.dump"

    start_time = time.time()
    df = load_dataframe_from_jsonl(test_file)
    print(df.head())
    end_time = time.time()
    print(end_time - start_time)
