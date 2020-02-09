import sys
sys.path.append('.')
sys.path.append('..')

from subreddit_frequency import load_dataframe_from_jsonl
from tqdm import tqdm
tqdm.pandas()

import seaborn as sns
from datetime import datetime
sns.set('paper')

from ipywidgets import interact
import pandas as pd
from pathlib import Path

comments_df = []
submissions_df = []

commments_df = pd.concat([
    load_dataframe_from_jsonl(a)
    for a in Path("data").glob("RC_*")
])
submissions_df = pd.concat([
    load_dataframe_from_jsonl(a)
    for a in Path("data").glob("RS_*")
])
#comments_df. load_dataframe_from_jsonl("data/RC_2018-11.zst_AmItheAsshole.dump")
#submissions_df. load_dataframe_from_jsonl("data/RS_2018-11.zst_AmItheAsshole.dump")
print(len(submissions_df))
print(len(comments_df))



def get_label_from_comments(df):
    try:
        return df[df.judgement != "UNK"].groupby('judgement').score.sum().idxmax()
    except ValueError:
        return "UNK"
    
def get_label_from_submission(submission_id):
    df = get_comments_from_id(comments_df, submission_id)
    return get_label_from_comments(df)
def get_comments_from_id(df, parent_id):
    cols = ['author_flair_text', 'stickied', 'author', 'body', 'score', 'score_abs', 'judgement']
    return df[df.prev_id == parent_id][cols]

def determine_AH(body):
    """Determines if poster thinks asshole or not asshole."""
    if body.startswith("YTA"):
        return "YTA"
    elif body.startswith("ESH"):
        return "ESH"
    elif body.startswith("NAH"):
        return "NAH"
    elif body.startswith("NTA"):
        return "NTA"
    else:
        return "UNK"

comments_df['prev_id'] = comments_df.parent_id.map(lambda x: x.split('_')[-1])
comments_df['score_abs'] = comments_df.score.map(abs)
comments_df['judgement'] = comments_df.body.map(determine_AH)
submissions_df['timestamp'] = submissions_df.created_utc.map(datetime.fromtimestamp)
submissions_df = submissions_df.sort_values('num_comments', ascending=False)


pd.set_option('display.max_rows', 500)
good_submissions_df = submissions_df.head(2000)

##display(list(good_submissions_df.head().title))
#display(list(good_submissions_df.head().id))



good_submissions_df['label'] = good_submissions_df.id.progress_map(get_label_from_submission)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
good_submissions_df[['title', 'label']].head()

dataset_df = good_submissions_df[good_submissions_df.label != 'UNK']
test_dataset_df = dataset_df.sample(frac=0.1)
traindev_dataset_df = dataset_df.drop(test_dataset_df.index)

train_dataset_df = traindev_dataset_df.sample(frac=0.8)
dev_dataset_df = traindev_dataset_df.drop(train_dataset_df.index)

dev_dataset_df.to_pickle('aita/aita-dev.pkl')
train_dataset_df.to_pickle('aita/aita-train.pkl')
test_dataset_df.to_pickle('aita/aita-test.pkl')