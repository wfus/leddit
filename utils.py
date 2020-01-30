from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd


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


