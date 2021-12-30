from logutil import *
from collections import defaultdict


class basefeaturegenerator:
    def __init__(self, data):
        self.data = data
        self.logger = logutil()
        self.generate_intermediate_data()

    def generate_intermediate_data(self):
        """生成中间feature数据，方便后续处理
        Args:

        Returns:

        Date:  2021/12/27
        """
        author_conferenceid_num = defaultdict(lambda: defaultdict(int))
        author_journalid_num = defaultdict(lambda: defaultdict(int))
        author_paper_num = defaultdict(int)  # 某个作者的论文数量
        paper_author_set = defaultdict(set)  # 这个论文的所有作者集合
        author_paper_coauthor_sum = defaultdict(int)

        for paperid, authorid, name, affiliation in self.data.paper_author_tuples:
            if paperid in self.data.paper_dict:
                cid = self.data.paper_dict[paperid]['conferenceid']
                jid = self.data.paper_dict[paperid]['journalid']
                if cid > 0:  # 说明是会议论文
                    author_conferenceid_num[authorid][cid] += 1
                if jid > 0:
                    author_journalid_num[authorid][jid] += 1
            author_paper_num[authorid] += 1
            paper_author_set[paperid].add(authorid)

        self.logger.debug("---获取这个作者和其它人员共同paper数---")
        author_coauthor_sumpapers = defaultdict(int)  # 这个作者和另一个作者合作次数dict[(id,id),cnt]
        author_paper_pair_set = set()  # set[(authorid, paperid)]
        temp_author_set = set()
        for (_authorid, _paperid, _label) in self.data.train_tuples + self.data.valid_tuples:
            coauthoers = paper_author_set[_paperid]
            author_paper_pair_set.add((_authorid, _paperid))
            for _coauthor in coauthoers:
                if _authorid != _coauthor:
                    author_coauthor_sumpapers[tuple(sorted([_authorid, _coauthor]))] = 0
                temp_author_set.add(_coauthor)
        self.logger.debug("---计算这个作者个其它人员共同paper数---")
        cnt = 0
        for _paper, authorset in paper_author_set.items():
            cnt += 1
            if cnt % 5000 == 0:
                self.logger.debug('count %d/%d done.' % (cnt, len(paper_author_set)))
            sorted_authorset = sorted(list(authorset))
            sorted_authorset = [author for author in sorted_authorset if author in temp_author_set]
            lens = len(sorted_authorset)
            for i in range(0, lens - 1):
                for j in range(i + 1, lens):
                    co_set = (sorted_authorset[i], sorted_authorset[j])
                    if co_set in author_coauthor_sumpapers:
                        author_coauthor_sumpapers[co_set] += 1
        self.logger.debug('---paper_author_set遍历结束---')

        for (_author, _paper) in author_paper_pair_set:
            for _coauthor in paper_author_set[_paper]:
                if _author != _coauthor:
                    co_set = tuple(sorted([_author, _coauthor]))
                    author_paper_coauthor_sum[(_author, _paper)] += author_coauthor_sumpapers[co_set]
        del author_coauthor_sumpapers  # 删除
        print("label-1")

        self.author_conferencid_num = author_conferenceid_num
        self.author_journalid_num = author_journalid_num
        self.author_paper_num = author_paper_num
        self.paper_author_set = paper_author_set
        self.author_paper_coauthor_sum = author_paper_coauthor_sum

    def get_feature(self, authorid, paperid):
        """
            获取(authorid, paperid)的基础特征
        Args:
            authorid, paperid, data
        Returns:
            list of features of (authorid, paperid)
            1) author在这个journal上发过的论文总数
            2）author在这个conference法国的论文总数
            3）这个author的论文总数
            4）这个论文的作者总数
            5）作者和coauthor的合作论文数
            6） 5)/4)
        Date:  2021/12/27
        """
        jid_nums = 0
        cid_nums = 0
        if paperid in self.data.paper_dict:
            jid = self.data.paper_dict[paperid]['journalid']
            cid = self.data.paper_dict[paperid]['conferenceid']
        else:
            jid = -1
            cid = -1
        if jid != -1:
            jid_nums = self.author_journalid_num[authorid][jid]
            cid_nums = self.author_conferencid_num[authorid][cid]
        try:
            author_papers_sum = self.author_paper_num[authorid]
        except AssertionError:
            author_papers_sum = 0
        try:
            paper_coauthor_num = len(self.paper_author_set[paperid])
        except AssertionError:
            paper_coauthor_num = 0
        try:
            author_coauthor_sum = self.author_paper_coauthor_sum[(authorid, paperid)]
        except AssertionError:
            author_coauthor_sum = 0
        try:
            ratio = float(author_coauthor_sum)/paper_coauthor_num
        except ZeroDivisionError:
            ratio = 0
        return [jid_nums, cid_nums, author_papers_sum, paper_coauthor_num, author_coauthor_sum, ratio]
