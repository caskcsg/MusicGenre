
import os
import json
from collections import defaultdict

filenames = os.listdir("./small/")
song2reviewer = defaultdict(list)
composer2style = defaultdict(list)
reviewer2style = defaultdict(list)

all_reviwer_name = []
reviewername_cnt = 0
for name in filenames:
    with open("./small/"+name, 'r', encoding='utf-8') as input:
        jobj = json.loads(input.readline().strip())
        reviewer_names = []

        composer = jobj["name"].split("-")[0].strip()
        composer2style[composer] += jobj['tags']

        for reviewobj in jobj["all_short_comments"]:
            song2reviewer[jobj["name"]].append(reviewobj["name"])
            reviewer2style[reviewobj["name"]] += jobj['tags']

        reviewer_names = set(reviewer_names)
        all_reviwer_name += reviewer_names
        reviewername_cnt += len(reviewer_names)

print(len(set(all_reviwer_name)))
print(reviewername_cnt)


def avg(dic):
    cnt = 0
    tmp = {}
    for k, v in dic.items():
        tmp[k] = set(v)
        cnt += len(set(v))
    print(cnt/len(tmp))

avg(composer2style)
avg(reviewer2style)

