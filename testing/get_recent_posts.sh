#!/bin/sh

wget -qO- http://www.reddit.com/r/AmItheAsshole/new/.json?count=5 \
| jq '.data.children[]|{title:.data.title,selftext:.data.selftext}' -c \
> data/aita_recent.jsonl
