#!/usr/bin/env python3

import praw
import argparse
import csv
import datetime
import requests
import time

"""
Running:
$ python3 reddit-api.py [path_to_input_file]

"""

# Globals
api_key_reddit = open('.api_key_reddit').read().strip()

# Starting up the CSV
current_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d-%Hh-%Mm'))  # was .strftime('%Y-%m-%d'))
csv_file_name = 'reddit/data/output/' + current_timestamp + '-reddit-comments.csv'
headers = ['thread_title', 'thread_link', 'created_utc',
           'author', 'body', 'score', 'ups', 'downs']
with open(csv_file_name, 'a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(headers)

# Making comment_count global to handle multiple store_csv() calls
comment_count = 0


def timestamp_to_utc(timestamp):
    """Convert unix timestamp to UTC date
    :param timestamp: int - the unix timestamp integer
    :return: utc_data - the date in YYYY-MM-DD format
    """
    timestamp = int(str(timestamp)[0:10])
    utc_date = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
    # utc_date = timestamp[0:10]
    return utc_date


def store_csv(thread_title, thread_link, comment):
    """Store the traffic stats as a CSV, with schema:
    > Output Schema: [date_time]-comments.csv
    :param video_id: string -
    :param video_title: string -
    :param comments: list -
    """
    global comment_count
    # for comment in comments:
    comment_count += 1
    row = [thread_title, thread_link,
           str(timestamp_to_utc(comment.created_utc)),
           str(comment_count)
           comment.author,
           comment.body,
           str(comment.score),
           str(comment.ups),
           str(comment.downs)]
    with open(csv_file_name, 'a') as csv_file1:
        csv_writer = csv.writer(csv_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

    with open(csv_file_name[:len(csv_file_name)-4] + "-" + thread_title + '.csv', 'a') as csv_file2:
        csv_writer = csv.writer(csv_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

    return ''


def traverse_branch(thread_title, thread_link, c1):
    # global comments
    store_csv(thread_title, thread_link, c1)
    # comments.append(c1.body)
    if c1.replies:
        for c2 in c1.replies:
            traverse_branch(thread_title, thread_link, c2)  # iteration


def prawler(thread_title, thread_link):
    rd = praw.Reddit(user_agent='Testing Comment Extraction (by /u/let-them-code-py)',
                     client_id='0ZyRF8JSRV1msA', client_secret=api_key_reddit)
                     # username='USERNAME', password='PASSWORD') - not needed for public comments
    # submission = rd.submission(url='https://www.reddit.com/r/politics/comments/54nrcs/
    # 2016_presidential_race_first_presidential/')
    submission = rd.submission(url=thread_link)
    submission.comments.replace_more(limit=0)

    all_comments = submission.comments.list()

    for c1 in submission.comments.list():
        traverse_branch(thread_title, thread_link, c1)
    
    return ''


def main(input_data):
    """Run top-level logic for API calls
    :param input_data: .txt - has schema: post_title, link
    """
    inputs = open(input_data).readlines()

    for thread in inputs:
        if thread.strip():  # if non-blank row; not enough to do "if var:"
            temp_reddit_link = thread.split(",")[1].strip()
            temp_reddit_title = thread.split(",")[0].strip()
            # Write headers for individual file
            with open(csv_file_name[:len(csv_file_name)-4] + "-" + temp_reddit_title + '.csv', 'a') as csv_file2:
                csv_writer = csv.writer(csv_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(headers)
            
            prawler(temp_reddit_title, temp_reddit_link)
            comment_count = 0
            # store_csv(temp_reddit_title, temp_reddit_link, comments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    args = parser.parse_args()
    main(args.input_data)
    # main()
