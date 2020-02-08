from subprocess import run
import shutil as sh

URL = "https://files.pushshift.io/reddit/comments"

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



def download_and_filter_by_sub(current_dump, subreddit):
    url = f"{URL}/{current_dump}"
    run(['wget', '-nc', url])

    # Now use the filter.sh command
    command = f"./filter.sh {current_dump} {subreddit}"
    print(f"Using command {command}")
    run(command, shell=True)

    sh.rmtree(current_dump)



for available_dump in available_dumps:
    download_and_filter_by_sub(available_dump, "AmItheAsshole")


