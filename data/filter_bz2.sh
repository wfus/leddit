#!/bin/bash
# Filters by subreddit of reddit dumps

DUMP_FILE=$1
SUBREDDIT=$2

if [ $# -lt 2 ]; then
	echo "Usage: ./filter.sh <dump_file> <subreddit>"
	exit
fi

echo $DUMP_FILE
echo $SUBREDDIT

pv $DUMP_FILE  | bzcat | jq --compact-output "select(.subreddit == \"${SUBREDDIT}\")" \
> ${DUMP_FILE}_${SUBREDDIT}.dump
