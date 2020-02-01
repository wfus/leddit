import pandas as pd
import json
from pathlib import Path
import argparse
from utils import load_dataframe_from_shard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Folder containing shareded comments',
        type=str)
    parser.add_argument('-t', '--topk', help='display topk subreddits with tag',
        default=20, type=int)
    args = parser.parse_args()

    filtered_shards = Path(args.folder)
    print(f"Looking in {filtered_shards}")
    shards = list(filtered_shards.glob('*'))
    print(f"Found {len(shards)} shards")
    
    df = load_dataframe_from_shard(filtered_shards)
    print(df.subreddit.value_counts().nlargest(args.topk))

            
