from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd


def load_dataframe_from_jsonl(fname):
    """Loads a reddit dump of a non-delimited jsonlist into a pandas dataframe.
    This won't work for files larger than a few gigabytes, the input needs to
    be filtered from the monthly dumps first."""
    dump_file = Path(fname)
    print(fname)
    comments = []
    with dump_file.open('r') as f:
        for line in tqdm(f):
            comment = json.loads(line.strip())
            comments.append(comment)

    df = pd.DataFrame(comments)
    return df
    

def load_dataframe_from_shard(shard_folder):
    """Loads a dataframe from a folder of files, each containing a 
    newline-delimited json list.
    
    Inputs:
        shard_folder [String]: path to folder containing the shards

    Output:
        df [pandas.DataFrame]: Dataframe made up of the jsonlists
    """
    filtered_shards = Path(shard_folder)
    shards = list(filtered_shards.glob('*'))

    comments = []
    for shard in shards:
        with shard.open('r') as f:
            for line in f:
                comment = json.loads(line)
                comments.append(comment)

    df = pd.DataFrame(comments)
    return df


