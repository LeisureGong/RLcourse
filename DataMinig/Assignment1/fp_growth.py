import copy

import pyfpgrowth
import sys
import pandas as pd

sys.setrecursionlimit(3000)


if __name__ == "__main__":
    path = "dataset/GroceryStore/Groceries.csv"
    source_data = pd.read_csv(path)
    dataset_list = source_data.values.tolist()
    dataset = list(map(lambda x: x[1].replace('{', '').replace('}', '').split(","), dataset_list))

    dataset1 = list()
    for i in range(0, 1):
        path = "dataset/UNIX_usage/USER" + str(i) + "/sanitized_all.981115184025"
        with open(path, 'r') as f:
            dataset_i = list()
            for line in f.readlines():
                line = line.strip('\n')
                if line == "**SOF**":
                    continue
                elif line == "**EOF**":
                    if len(dataset_i) > 0:
                        tmp = copy.deepcopy(dataset_i)
                        dataset1.append(tmp)
                        dataset_i.clear()
                else:
                    dataset_i.append(line)

    patterns = pyfpgrowth.find_frequent_patterns(dataset, 10)

    rules = pyfpgrowth.generate_association_rules(patterns, 0.5)

    print(patterns)
    print()
    print(rules)
