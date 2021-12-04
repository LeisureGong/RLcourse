from efficient_apriori import apriori
import pandas as pd

if __name__ == "__main__":
    path = "dataset/GroceryStore/Groceries.csv"
    source_data = pd.read_csv(path)
    dataset_list = source_data.values.tolist()
    dataset = list(map(lambda x: x[1].replace('{', '').replace('}', '').split(","), dataset_list))

    transactions = [tuple(row) for row in dataset]

    itemsets, rules = apriori(transactions, min_support=0.01, min_confidence=0.5)
    print(rules)