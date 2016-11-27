#!/usr/bin/env python3

import argparse
import csv
import datetime
import requests
import time

"""
Running:
$ python3 youtube-api.py [path_to_input_file]

"""

# Globals
api_key = open('.api_key').read().strip()
quota_counter = 100000

# Starting up the CSV
current_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d-%Hh-%Mm'))  # was .strftime('%Y-%m-%d'))
csv_file_name = 'data/output/' + current_timestamp + '-youtube-comments.csv'
headers = ['video_title', 'video_id',
           'comment_id', 'comment_date', 'updated_date', 'commenter_url', 'commenter_name', 'like_count',
           'top_level_comment_num', 'top_level_comment', 'comment_reply']
with open(csv_file_name, 'a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(headers)

# Making comment_count global to handle multiple store_csv() calls
comment_count = 0


def store_csv(video_id, video_title, comments):
    """Store the traffic stats as a CSV, with schema:
    > Output Schema: [date_time]-comments.csv
    video_id, comment_id, comment_date, updated_date, commenter_name, parent_comment, child_comment,
    :param video_id: string -
    :param video_title: string -
    :param comments: list -
    """
    global comment_count
    for comment in comments:
        comment_count += 1
        row = [video_title, video_id,
               comment['id'],
               comment['snippet']['topLevelComment']['snippet']['publishedAt'],
               comment['snippet']['topLevelComment']['snippet']['updatedAt'],
               comment['snippet']['topLevelComment']['snippet']['authorChannelUrl'],
               comment['snippet']['topLevelComment']['snippet']['authorDisplayName'],
               comment['snippet']['topLevelComment']['snippet']['likeCount'],
               str(comment_count),
               comment['snippet']['topLevelComment']['snippet']['textDisplay'],
               " - "]
        with open(csv_file_name, 'a') as csv_file1:
            csv_writer = csv.writer(csv_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)

        with open(csv_file_name[:len(csv_file_name)-4] + "-" + video_id + '.csv', 'a') as csv_file2:
            csv_writer = csv.writer(csv_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)

        # Getting the replies to each topLevelComment
        if comment.get('replies'):
            for reply in comment['replies']['comments']:
                row = [video_title, video_id,
                       reply['id'],
                       reply['snippet']['publishedAt'],
                       reply['snippet']['updatedAt'],
                       reply['snippet']['authorChannelUrl'],
                       reply['snippet']['authorDisplayName'],
                       reply['snippet']['likeCount'],
                       " - ",
                       " - ",
                       reply['snippet']['textDisplay']]

                with open(csv_file_name, 'a') as csv_file1:
                    csv_writer = csv.writer(csv_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

                with open(csv_file_name[:len(csv_file_name)-4] + "-" + video_id + '.csv', 'a') as csv_file2:
                    csv_writer = csv.writer(csv_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
    return ''


def send_request(resource, query_volume, video_id, video_title, part, max_results, order_by):
    """
    :param resource: 'commentThreads' - only this for now
    :param query_volume: 'all' or 'once' - set to 'all' to get all comments
    :param video_id:
    :param video_title:
    :param part:
    :param max_results:
    :param order_by:
    :return:
    """
    if resource == 'commentThreads':
        # GET https://www.googleapis.com/youtube/v3/commentThreads?
        # ^ from https://developers.google.com/youtube/v3/guides/implementation/comments
        base_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
        payload = {
            'part': part,
            'maxResults': max_results,
            'order': order_by,
            'videoId': video_id,
            'key': api_key
        }
        response = requests.get(base_url, params=payload)
        response_json = response.json()
        comments_list = []

        if query_volume == 'once' or response_json.get('nextPageToken') is None:
            comments_list += response_json['items']
            store_csv(video_id, video_title, comments_list)
            return comments_list

        if query_volume == 'all':
            comments_list += response_json['items']
            print(len(comments_list))  # Debug
            store_csv(video_id, video_title, comments_list)
            comments_list = []
            # Getting further comments as long as there's a paging token
            while response_json.get('nextPageToken'):
                next_page_token = response_json.get('nextPageToken')
                if next_page_token:
                    print('nextPageToken: ' + next_page_token[:20])  # Debug
                elif next_page_token is None:
                    print('nextPageToken: None')
                payload = {
                    'part': part,
                    'maxResults': max_results,
                    'order': order_by,
                    'videoId': video_id,
                    'pageToken': next_page_token,
                    'key': api_key
                }
                response_json = requests.get(base_url, params=payload).json()
                comments_list += response_json['items']
                print(len(comments_list))  # Debug
                store_csv(video_id, video_title, comments_list)
                time.sleep(1)
                comments_list = []
            return comments_list


def main(input_data):
    """Run top-level logic for API calls
    :param input_data: .txt - has schema: video_title, video_id
    """
    global comment_count

    inputs = open(input_data).readlines()  # each row = video_title, video_link/_id

    for video in inputs:
        if video.strip():  # if non-blank row; not enough to do "if var:"
            temp_video_link = video.split(",")[1].strip().split('https://www.youtube.com/watch?v=')
            # above 'https://' split() will return full string if can't split, so get the last item in split list
            temp_video_id = temp_video_link[len(temp_video_link)-1]
            temp_video_title = video.split(",")[0].strip()
            comments_list = send_request(resource='commentThreads',
                                         query_volume='all',
                                         video_id=temp_video_id,
                                         video_title=temp_video_title,
                                         part='snippet,replies',
                                         max_results=100,
                                         order_by='relevance')  # 'time' or 'relevance'
            print(temp_video_title + ": " + str(len(comments_list)) + ", " + str(comment_count) + " comments.")
            time.sleep(10)
            comment_count = 0

    print("Done main().")
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    args = parser.parse_args()
    main(args.input_data)
    # main()


"""
* * Sample request:

GET https://www.googleapis.com/youtube/v3/commentThreads?
part=snippet%2Creplies
&maxResults=100
&order=relevance
&pageToken=[long_text_string]
&videoId=srXsCRnSgBA
&key={YOUR_API_KEY}

* * Response structure:

{
 "kind": "youtube#commentThreadListResponse",
 "etag": "\"I_8xdZu766_FSaexEaDXTIfEWc0/IiDEZQ7_qegXjFH-cMADT2bWo0s\"",
 "nextPageToken": '...'
 "pageInfo": {
  "totalResults": 100,
  "resultsPerPage": 100
 },
 "items": [
  {
   "kind": "youtube#commentThread",
   "etag": "\"I_8xdZu766_FSaexEaDXTIfEWc0/W5uUBQboigHpYX84pcPhDHnm46g\"",
   "id": "z12tfd0wpun0ufjmv22zxpkg3pujhf43x",
   "snippet": {
    "videoId": "srXsCRnSgBA",
    "topLevelComment": {
     "kind": "youtube#comment",
     "etag": "\"I_8xdZu766_FSaexEaDXTIfEWc0/qSRyuWb813c3YJG_X10Tmw-13rc\"",
     "id": "z12tfd0wpun0ufjmv22zxpkg3pujhf43x",
     "snippet": {
      "authorDisplayName": "euphtygrit",
      "authorProfileImageUrl": "https://lh6.googleusercontent.com/...",
      "authorChannelUrl": "http://www.youtube.com/channel/UCYFydhhHbFNXnSGwjhGucGA",
      "authorChannelId": {
       "value": "UCYFydhhHbFNXnSGwjhGucGA"
      },
      "videoId": "srXsCRnSgBA",
      "textDisplay": "...\ufeff",
      "canRate": true,
      "viewerRating": "none",
      "likeCount": 121,
      "publishedAt": "2016-07-02T16:00:19.000Z",
      "updatedAt": "2016-07-02T16:00:19.000Z"
     }
    },
    "canReply": true,
    "totalReplyCount": 5,
    "isPublic": true
   },
   "replies": {
    "comments": [
     {
      "kind": "youtube#comment",
      "etag": "\"I_8xdZu766_FSaexEaDXTIfEWc0/btEd5iIPkf75Yj6LXsVAXP2iLdA\"",
      "id": "z12tfd0wpun0ufjmv22zxpkg3pujhf43x.1473421938255174",
      "snippet": {
       "authorDisplayName": "박지열",
       "authorProfileImageUrl": "https://lh4.googleusercontent.com/...",
       "authorChannelUrl": "http://www.youtube.com/channel/UCxK7Lmmsa-evY3ZM61dpXmg",
       "authorChannelId": {
        "value": "UCxK7Lmmsa-evY3ZM61dpXmg"
       },
       "videoId": "srXsCRnSgBA",
       "textDisplay": "...",
       "parentId": "z12tfd0wpun0ufjmv22zxpkg3pujhf43x",
       "canRate": true,
       "viewerRating": "none",
       "likeCount": 2,
       "publishedAt": "2016-09-09T11:52:18.000Z",
       "updatedAt": "2016-09-09T11:52:18.000Z"
      }
     },
     {...more replies...}
     }
    ]
   }
  }, ...
}
"""
