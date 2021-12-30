import csv
import pandas as pd
from collections import defaultdict


class Data:
    def __init__(self, train_csv, valid_csv, info_input):
        self.info_input = info_input
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.paper_author_csv = info_input + '/PaperAuthor.csv'
        self.author_csv = info_input + '/Author.csv'
        self.paper_csv = info_input + '/Paper.csv'
        self.conference_csv = info_input + 'Conference.csv'
        self.Journal_csv = info_input + '/Journal.csv'
        self.train_tuples = self.read_train_csv(self.train_csv)
        self.valid_tuples = self.read_valid_csv(self.valid_csv)
        self.paper_author_tuples = self.read_paperauthor_csv(self.paper_author_csv)
        self.paper_dict = self.read_paper_csv(self.paper_csv)
        self.author_dict = self.read_author_csv(self.author_csv)

    # return (authorid, paperid, label)
    def read_train_csv(self, train_csv):
        train_tuples = list()
        # train_data = pd.read_csv(train_csv)x
        train_data_csv = open(train_csv, 'r')
        train_data = csv.reader(train_data_csv)
        next(train_data)
        for cols in train_data:
            authorid = int(cols[0])
            confirmPapers = map(lambda x: int(x), cols[1].split())
            deletedPaper = map(lambda x: int(x), cols[2].split())
            for paper_id in confirmPapers:
                train_tuples.append([authorid, paper_id, 1])
            for paper_id in deletedPaper:
                train_tuples.append([authorid, paper_id, 0])
        return train_tuples

    def read_valid_csv(self, valid_csv):
        """读取valid或test数据集合
        Args:
            valid_csv: paper_id, author_id
        Returns:

        Date:  2021/12/27
        """
        valid_tuples = list()
        valid_data_csv = open(valid_csv, 'r')
        valid_data = csv.reader(valid_data_csv)
        next(valid_data)
        for cols in valid_data:
            author_id = int(cols[1])
            paper_id = int(cols[2])
            valid_tuples.append([author_id, paper_id, 0])
        return valid_tuples

    def read_paper_csv(self, paper_csv):
        """读取Paper.csv

        Args:
            Paper.csv
        Returns:
            dict: [paper_id,('conferenceid', 'journalid', 'title', 'year', 'keyword')]
        Date:  2021/12/27
        """
        paper_dict = defaultdict(lambda: dict())
        # paper_data = pd.read_csv(paper_csv)
        paper_data_csv = open(paper_csv, 'r')
        paper_data = csv.reader(paper_data_csv)
        next(paper_data)
        # times = 0
        for cols in paper_data:
            paper_id = int(cols[0])
            conference_id = int(cols[3])
            journal_id = int(cols[4])
            paper_dict[paper_id]['conferenceid'] = conference_id
            paper_dict[paper_id]['journalid'] = journal_id
            paper_dict[paper_id]['title'] = cols[1]
            paper_dict[paper_id]['year'] = int(cols[2])
            paper_dict[paper_id]['keyword'] = cols[5]
            # times += 1
            # if times > 200001:
            #     break
        return paper_dict

    def read_author_csv(self, author_csv):
        """

        Args:

        Returns:
             dict of author('name', 'affiliation')
        Date:  2021/12/27
        """
        author_dict = defaultdict(lambda: dict())
        # author_data = pd.read_csv(author_csv)
        author_data_csv = open(author_csv, 'r')
        author_data = csv.reader(author_data_csv)
        next(author_data)
        for cols in author_data:
            authorid = int(cols[0])
            name = cols[1]
            affi = cols[2]
            author_dict[authorid]['name'] = name
            author_dict[authorid]['affi'] = affi
        return author_dict

    def read_paperauthor_csv(self, paperauthor_csv):
        """ 存放(paper_id, author_id, name, affiliation)
        Args:

        Returns:

        Date:  2021/12/27
        """
        # paperauthor_data = pd.read_csv(paperauthor_csv)
        paperauthor_data_csv = open(paperauthor_csv, 'r')
        paperauthor_data = csv.reader(paperauthor_data_csv)
        next(paperauthor_data)
        paperauthor_tuples = list()
        # times = 0
        for cols in paperauthor_data:
            paper_id = int(cols[0])
            author_id = int(cols[1])
            name = cols[2]
            affi = cols[3]
            paperauthor_tuples.append((paper_id, author_id, name, affi))
            # times += 1
            # if times > 200001:
            #     break
        return paperauthor_tuples

    def read_conference_jorunal_csv(self, conference_csv):
        cid_info_dict = defaultdict(lambda: defaultdict())
        # cid_data = pd.read_csv(conference_csv)
        cid_data_csv = open(conference_csv, 'r')
        cid_data = csv.reader(cid_data_csv)
        next(cid_data)
        for cols in cid_data:
            cid = int(cols[0])
            cid_info_dict[cid]['shortname'] = cols[1]
            cid_info_dict[cid]['longname'] = cols[2]
        return cid_info_dict
