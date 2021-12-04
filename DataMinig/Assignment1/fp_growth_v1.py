import itertools
import pandas as pd


class TreeNode(object):
    """
    FP树节点的定义
    """
    def __init__(self, value, count, parent):
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        检查孩子节点中是否包含某个value
        """
        for node in self.children:
            if node.value == value:
                return True
        return False

    def get_child(self, value):
        for node in self.children:
            if node.value == value:
                return node
        return None

    def add_child(self, value):
        """
        添加孩子节点
        """
        child = TreeNode(value, 1, self)
        self.children.append(child)
        return child

class FPTree(object):
    def __init__(self, transactions ,threshold, root_value, root_count):
        """
        :param threshold:
        :param root_value:
        :param root_count:
        """
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(transactions, root_value, root_count, self.frequent, self.headers)

    def find_frequent_items(transactions, threshold):
        """
        创建一个大于或等于threshold的dict
        """
        items = {}
        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1
        for key in list(items.keys()):
            if items[key] < threshold:
                del items[key]
        return items

    def build_header_table(frequent):
        # 创建header table
        headers = {}
        for key in frequent.keys():
            headers[key] = None
        return headers

    def build_fptree(self, transactions, root_val, root_cnt, frequent, headers):
        root = TreeNode(root_val, root_cnt, None)
        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) > 0:
                # todo
                self.insert_tree(sorted_items, root, headers)
        return root

    def insert_tree(self, items, node, headers):
        # 在FP树中插入节点
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            # 创建新节点
            child = node.add_child(first)
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child
        # 递归地构建tree
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)

    def tree_has_single_path(self, node):
        # todo 检测
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):
        # 搜索频繁项集
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):
        #
        suffix = self.root.value
        if suffix is not None:
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]
            return new_patterns
        return patterns

    def generate_pattern_list(self):
        # 生成frequent items
        patterns = {}
        items = self.frequent.keys()

        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items)+1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.frequent[x] for x in subset])
        return patterns

    def mine_sub_trees(self, threshold):
        """
        生成子树并挖掘频发项集
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])
        # 逆转
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]

            # 获取某个item的全部项
            while node is not None:
                suffixes.append(node)
                node = node.link

            # 对于某个item， 记录从该节点到root节点的路径
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # 挖掘频繁项集
            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # 把子树的频繁项集插入到顶部树中
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]
        return patterns


def find_frequent_patterns(transactions, support_threshold):
    tree = FPTree(transactions, support_threshold, None, None)
    return tree.mine_patterns(support_threshold)

def generate_association_rules(patterns, confidence_threshold):
    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support
                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)
    return rules

if __name__ == "__main__":
    path = "dataset/GroceryStore/Groceries.csv"
    source_data = pd.read_csv(path)
    dataset_list = source_data.values.tolist()
    dataset = list(map(lambda x: x[1].replace('{', '').replace('}', '').split(","), dataset_list))

    patterns = find_frequent_patterns(dataset, 50)

    rules = generate_association_rules(patterns, 0.7)

    print(patterns)
    print()
    print(rules)
