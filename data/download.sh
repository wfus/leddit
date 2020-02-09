#!/bin/sh
# The urls are in this format
# https://files.pushshift.io/reddit/comments/RC_2018-10.zst
# https://files.pushshift.io/reddit/comments/RC_2015-02.bz2
BASE_URL=https://files.pushshift.io/reddit/comments
FILE=RC_2018-11.zst

wget $BASE_URL/$FILE


