from pathlib import Path
import json
from tqdm import tqdm
from multiprocessing import Pool
import argparse


def process_file(batch):
    """Takes in a tuple of input file and output file. Processing input file
    by applying a filter and then writes this to the output file.
    """
    reddit_file, output_file = batch
    with reddit_file.open('r') as f:
        with output_file.open('w') as out_f:
            for line in f:
                try:
                    comment = json.loads(line.strip())
                    lowercase_text = comment['body'].lower()
                    if any([keyword in lowercase_text for keyword in KEYWORDS]):
                        out_f.write(line)
                except json.decoder.JSONDecodeError:
                    # Some lines may be split differently
                    pass


if __name__ == '__main__':
    REDDIT_DUMP_FOLDER = 'RC_2019-09_shard'
    KEYWORDS = ['tradwife']
    reddit_folder = Path(REDDIT_DUMP_FOLDER)
    output_folder = Path(REDDIT_DUMP_FOLDER + "_tradwife")
    output_folder.mkdir(exist_ok=True, parents=True)

    processing = []
    for shard_path in reddit_folder.glob('*'):
        output_file = output_folder / shard_path.name
        processing.append((shard_path, output_file))
    num_items = len(processing)
    print(f"Number of shards to process: {num_items}")

    with Pool(20) as p:
        _ = list(
            tqdm(
                p.imap_unordered(process_file, processing), total=num_items
            )
        )
