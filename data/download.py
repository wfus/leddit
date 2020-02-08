from subprocess import run
import shutil as sh
import os
from pathlib import Path

URL = "https://files.pushshift.io/reddit/comments"
SUBMISSIONS_URL = "https://files.pushshift.io/reddit/submissions"

available_dumps = [
    'RC_2018-10.zst',
    'RC_2018-11.zst',
    'RC_2018-12.zst',
    'RC_2019-01.zst',
    'RC_2019-02.zst',
    'RC_2019-03.zst',
    'RC_2019-04.zst',
    'RC_2019-05.zst',
    'RC_2019-06.zst',
    'RC_2019-07.zst',
    'RC_2019-08.zst',
    'RC_2019-09.zst',
]

available_submission_dumps = [
    "RS_2018-11.zst",
    "RS_2018-12.zst",
    "RS_2019-01.zst",
    "RS_2019-02.zst",
    "RS_2019-03.zst",
    "RS_2019-04.zst",
    "RS_2019-05.zst",
    "RS_2019-06.zst",
    "RS_2019-07.zst",
    "RS_2019-08.zst",
    "RS_2019-09.zst",
]


def download_submissions_and_filter_by_sub(current_dump, subreddit):
    output_fname = f"{current_dump}_{subreddit}.dump"
    if not Path(output_fname).exists():
        url = f"{SUBMISSIONS_URL}/{current_dump}"
        run(['wget', '-nc', url])

        # Now use the filter.sh command
        command = f"./filter.sh {current_dump} {subreddit}"
        print(f"Using command {command}")
        run(command, shell=True)
    else:
        print(f"Already see output file at {output_fname}")

    try:
        os.remove(current_dump)
    except OSError:
        pass


def download_and_filter_by_sub(current_dump, subreddit):
    output_fname = f"{current_dump}_{subreddit}.dump"
    if not Path(output_fname).exists():
        url = f"{URL}/{current_dump}"
        run(['wget', '-nc', url])

        # Now use the filter.sh command
        command = f"./filter.sh {current_dump} {subreddit}"
        print(f"Using command {command}")
        run(command, shell=True)
    else:
        print(f"Already see output file at {output_fname}")


    try:
        os.remove(current_dump)
    except OSError:
        pass

# for available_dump in available_dumps:
#     download_and_filter_by_sub(available_dump, "AmItheAsshole")


for available_dump in available_submission_dumps:
    download_submissions_and_filter_by_sub(available_dump, "AmItheAsshole")


