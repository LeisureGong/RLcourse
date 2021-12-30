from logutil import *
from collections import defaultdict as ddict

'''
Description: The feature Generator
feature1: trick1
feature2: author 在 this paper发表的年份曾经发表的论文个数
feature3: 作者发表在journal的论文个数
feature4: 作者发表在conference的论文个数
feature5: <authorid jid>个数/authorid journal的个数
feature6: <authorid cid>个数/authorid conference的个数
'''


class featuregenerator01:
    def __init__(self, data):
        self.data = data
        self.logger = logutil()
        self.generate_intermediate_data()

    '''
    Description: Generate the Intermedia feature data 
    Parameter: None
    Return: None
    '''

    def generate_intermediate_data(self):
        authorid_year_num = ddict(lambda: ddict(int))
        authorid_cid_num = ddict(lambda: ddict(int))
        authorid_jid_num = ddict(lambda: ddict(int))
        authorid_conference_sum = ddict(int)
        authorid_journal_sum = ddict(int)
        authorid_paperid_sum = dict()

        cal_authorid_set = set()
        for authorid, paperid, rate in self.data.train_tuples:
            authorid_paperid_sum[(authorid, paperid)] = 0
            cal_authorid_set.add(authorid)

        for paperid, authorid, name, affi in self.data.paper_author_tuples:
            paperid = int(paperid)
            authorid = int(authorid)
            if (authorid, paperid) in authorid_paperid_sum:
                authorid_paperid_sum[(authorid, paperid)] += 1

            if authorid in cal_authorid_set:
                if paperid in self.data.paper_dict:
                    year = self.data.paper_dict[paperid]['year']
                    if year != 0:
                        authorid_year_num[authorid][year] += 1
                    cid = self.data.paper_dict[paperid]['conferenceid']
                    if cid != 0:
                        authorid_cid_num[authorid][cid] += 1
                        authorid_conference_sum[authorid] += 1
                    jid = self.data.paper_dict[paperid]['journalid']
                    if jid != 0:
                        authorid_jid_num[authorid][jid] += 1
                        authorid_journal_sum[authorid] += 1

        self.authorid_year_num = authorid_year_num
        self.authorid_cid_num = authorid_cid_num
        self.authorid_jid_num = authorid_jid_num
        self.authorid_conference_sum = authorid_conference_sum
        self.authorid_journal_sum = authorid_journal_sum
        self.authorid_paperid_sum = authorid_paperid_sum

    def get_feature(self, authorid, paperid):
        year_sum = 0
        aid_cid_sum = 0
        aid_jid_sum = 0
        conference_sum = self.authorid_conference_sum[authorid]
        journal_sum = self.authorid_journal_sum[authorid]
        if paperid in self.data.paper_dict:
            yearid = self.data.paper_dict[paperid]['year']
            year_sum = self.authorid_year_num[authorid][yearid]

            cid = self.data.paper_dict[paperid]['conferenceid']

            if cid != 0:
                aid_cid_sum = self.authorid_cid_num[authorid][cid]
            jid = self.data.paper_dict[paperid]['journalid']

            if jid != 0:
                aid_jid_sum = self.authorid_jid_num[authorid][jid]

            aid_cid_conference_ratio = 0
            if conference_sum > 0:
                aid_cid_conference_ratio = float(aid_cid_sum) / conference_sum
            aid_jid_journal_ratio = 0
            if journal_sum > 0:
                aid_jid_journal_ratio = float(aid_jid_sum) / journal_sum

        return [self.authorid_paperid_sum.get((authorid, paperid), 0), year_sum, conference_sum, journal_sum]
