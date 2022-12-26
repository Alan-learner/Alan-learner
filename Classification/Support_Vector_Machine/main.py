# encoding: utf-8
# author: Alan-learner

import matplotlib.pyplot as plt

from sklearn.svm import SVC  # 分类算法
from sklearn import datasets


def create_data():
    X, y = datasets.make_blobs(n_samples=100,  # 样本量
                               n_features=2,  # 二维数据，便于画图展示
                               centers=2,  # 两类
                               random_state=3)  # 随机数状态，固定了
    plt.scatter(X[:, 0], X[:, 1], c=y)
    return X, y


def fit_model(X, y):
    svc = SVC(kernel='linear')  # kernel 表示核函数：linear，线性
    svc.fit(X, y)
    return svc


def get_para(model):
    w_ = model.coef_
    b_ = model.intercept_
    w = - w_[0, 0] / w_[0, 1]
    b, = - b_ / w_[0, 1]
    return w, b


def draw_graph(model, w, b, X, y):
    sv = model.support_vectors_
    x = np.linspace(-5, 1, 100)

    y_result = w * x + b

    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.plot(x, y_result, color='red')

    # 上边界和下边界
    b1 = sv[0][1] - w * sv[0][0]
    plt.plot(x, w * x + b1, color='blue', ls='--')

    b2 = sv[-1][1] - w * sv[-1][0]
    plt.plot(x, w * x + b2, color='black', ls='--')


def main():
    X, y = create_data()
    svc = fit_model(X, y)
    w, b = get_para(model=svc)
    draw_graph(svc, w, b, X, y)


if __name__ == '__main__':
    main()
