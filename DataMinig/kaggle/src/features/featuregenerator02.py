from collections import defaultdict as ddict

'''
Description: add features as list
1) 添加paperlist中出现该paper的个数, 0.004+
2) coauthor sumpapers/<paper author num>, 0.00099+, 这个feature加在了BaselineFeatureGenerator中
'''


class featuregenerator02:
    def __init__(self, data):
        self.data = data
        self.get_intermediate_data()

    def get_intermediate_data(self):
        author_paper_cocur_nums = ddict(int)
        for (author, paperid, label) in self.data.train_tuples:
            author_paper_cocur_nums[(author, paperid)] += 1
        self.author_paper_cocur_nums = author_paper_cocur_nums

    def get_feature(self, authorid, paperid):
        return [self.author_paper_cocur_nums[(authorid, paperid)]]
