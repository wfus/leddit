from subprocess import run
import shutil as sh
import os
from pathlib import Path
from multiprocessing import Pool

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
    "RC_2017-11.bz2",
    "RC_2017-10.bz2",
    "RC_2017-09.bz2",
    "RC_2017-08.bz2",
    "RC_2017-07.bz2",
    "RC_2017-06.bz2",
    "RC_2017-05.bz2",
    "RC_2017-04.bz2",
    "RC_2017-03.bz2",
    "RC_2017-02.bz2",
    "RC_2017-01.bz2",
    "RC_2016-12.bz2",
    "RC_2016-11.bz2",
    "RC_2016-10.bz2",
    "RC_2016-09.bz2",
    "RC_2016-08.bz2",
    "RC_2016-07.bz2",
    "RC_2016-06.bz2",
    "RC_2016-05.bz2",
    "RC_2016-04.bz2",
    "RC_2016-03.bz2",
    "RC_2016-02.bz2",
    "RC_2016-01.bz2",
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
    "RS_2017-11.bz2",
    "RS_2017-10.bz2",
    "RS_2017-09.bz2",
    "RS_2017-08.bz2",
    "RS_2017-07.bz2",
    "RS_2017-06.bz2",
    "RS_2017-05.bz2",
    "RS_2017-04.bz2",
    "RS_2017-03.bz2",
    "RS_2017-02.bz2",
    "RS_2017-01.bz2",
    "RS_2016-01.zst",
    "RS_2016-02.zst",
    "RS_2016-03.zst",
    "RS_2016-04.zst",
    "RS_2016-05.zst",
    "RS_2016-06.zst",
    "RS_2016-07.zst",
    "RS_2016-08.zst",
    "RS_2016-09.zst",
    "RS_2016-10.zst",
    "RS_2016-11.zst",
    "RS_2016-12.zst",
]

def filter_raw_dump(current_dump, subreddit):
    if current_dump.endswith('.bz2'):
        print(f"Running filter command for .bz2 dump {current_dump}")
        command = f"./filter_bz2.sh {current_dump} {subreddit}"
    else:
        print(f"Running filter command for .zst dump {current_dump}") 
        command = f"./filter.sh {current_dump} {subreddit}"
    return run(command, shell=True)


def download_and_filter_by_sub(current_dump, subreddit, base_url):
    output_fname = f"{current_dump}_{subreddit}.dump"
    if Path(output_fname).exists():
        print(f"Already see output file at {output_fname}")
        filter_status = 0
    else:
        url = f"{base_url}/{current_dump}"
        if not Path(current_dump).exists():
            print(f'Downloading {current_dump}')
            run(['wget', '-nc', url])
        else:
            print(f'{current_dump} already downloaded')

        filter_status = filter_raw_dump(current_dump, subreddit).returncode

    try:
        if filter_status == 0:
            os.remove(current_dump)
        else:
            print(f'Not removing {current_dump} because of an error')
    except OSError:
        pass

if __name__ == '__main__':
    sub_name = "AmItheAsshole"
    def download_function(x):
        return download_and_filter_by_sub(x, sub_name, SUBMISSIONS_URL)
    with Pool(3) as p:
        p.map(download_function, available_submission_dumps)
    def download_function(x):
        return download_and_filter_by_sub(x, sub_name, URL)
    with Pool(3) as p:
        p.map(download_function, available_dumps)
