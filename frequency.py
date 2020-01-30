import pandas as pd
import json
from pathlib import Path
import argparse


def load_dataframe_from_shard(shard_folder):
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

            
