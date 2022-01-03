from collections import defaultdict as ddict
import re

"""
keyword特征
"""


class featuregenerator02:
    def __init__(self, data):
        self.data = data
        self.get_intermediate_data()

    def get_intermediate_data(self):
        paper_keywords = ddict(set)
        for paperid, info_dict in self.data.paper_info_dict.items():
            keywords = split_keywords(info_dict['keyword'])
            for k in keywords:
                paper_keywords[paperid].add(k)
        self.paper_keywords = paper_keywords

        author_keywords = ddict(lambda: ddict(int))
        need_authors = get_need_authors(self.data)
        for (paperid, authorid, name, affi) in self.data.paperauthor_tuples:
            if authorid in need_authors and paperid in self.data.paper_info_dict:
                keywords = split_keywords(self.data.paper_info_dict[paperid]['keyword'])
                for k in set(keywords):
                    author_keywords[authorid][k] += 1
        self.author_keywords = author_keywords
        return


def get_feature(self, authorid, paperid):
    paper_kws = self.paper_keywords[paperid]
    author_kw_counts = self.author_keywords[authorid]

    match_num = 0
    for kw in paper_kws:
        if author_kw_counts[kw] > 1:
            match_num += 1

    try:
        return [match_num, len(paper_kws), float(match_num) / len(paper_kws)]
    except:
        return [0, 0, 0]


def split_keywords(keyword):
    return [w for w in re.split(r',|\|| |;', keyword) if w != '']


def get_need_authors(data):
    need_authors = set()
    for (authorid, paperid, label) in data.train_tuples + data.valid_tuples:
        need_authors.add(authorid)
    return need_authors
