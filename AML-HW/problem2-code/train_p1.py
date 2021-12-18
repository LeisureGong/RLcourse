import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

from load_data import load_images


def train(algo, pca_dim=None):
    # load data
    train_xs, train_ys = load_images("train")
    test_xs, test_ys = load_images("test")

    train_xs = train_xs.reshape(train_xs.shape[0], -1)
    test_xs = test_xs.reshape(test_xs.shape[0], -1)

    n_tr = len(train_xs)

    # PCA
    if pca_dim is not None:
        xs = np.concatenate([train_xs, test_xs], axis=0)
        pca = PCA(n_components=pca_dim)
        xs = pca.fit_transform(xs)
        train_xs, test_xs = xs[0:n_tr], xs[n_tr:]

    if algo == "LR":
        model = LogisticRegression(
            multi_class="multinomial", C=10.0,
            solver="lbfgs", max_iter=5000
        )
    elif algo == "DT":
        model = DecisionTreeClassifier(
            max_depth=10
        )
    elif algo == "RF":
        model = RandomForestClassifier(
            n_estimators=500
        )
    elif algo == "NN":
        model = KNeighborsClassifier(
            n_neighbors=20
        )

    model.fit(train_xs, train_ys)
    pred_train_ys = model.predict(train_xs)
    pred_test_ys = model.predict(test_xs)

    train_acc = np.mean(train_ys == pred_train_ys)
    test_acc = np.mean(test_ys == pred_test_ys)

    print("[{},{}] Train Acc:{:.5f}, Test Acc:{:.5f}".format(
        algo, pca_dim, train_acc, test_acc
    ))
    return train_acc, test_acc


if __name__ == "__main__":
    train_accs = []
    test_accs = []
    for algo in ["LR", "DT", "RF", "NN"]:
        for pca_dim in [None, 1]:
            train_acc, test_acc = train(algo, pca_dim)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
    train_accs = np.array(train_accs).reshape(4, 2)
    test_accs = np.array(test_accs).reshape(4, 2)

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.imshow(train_accs)

    for i in range(2):
        for j in range(4):
            ax.text(i, j, "{:.1f}".format(train_accs[j][i] * 100.0))

    ax.set_xticks(range(2))
    ax.set_xticklabels(["None", "1"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["LR", "DT", "RF","NN"])
    ax.set_title("Train Acc")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.imshow(test_accs)

    for i in range(2):
        for j in range(4):
            ax.text(i, j, "{:.1f}".format(test_accs[j][i] * 100.0))

    ax.set_xticks(range(2))
    ax.set_xticklabels(["None", "1"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["LR", "DT", "RF", "NN"])
    ax.set_title("Test Acc")

    fig.tight_layout()
    fig.savefig("./trainp0.jpg")
    plt.show()
