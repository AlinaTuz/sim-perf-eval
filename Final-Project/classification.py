import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from time import time

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

# Inspired by:
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# Initializing some clasification algorithms

names = [
    "Decision Tree",
    "Random Forest",
    "Linear Support Vector Machine",
    "RBF Support Vector Machine",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "Neural Network"
]

classifiers = [ # The following parameters may need to be fine-tuned
    sklearn.tree.DecisionTreeClassifier(max_depth=5, random_state=44),
    sklearn.ensemble.RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=44
    ),
    sklearn.svm.SVC(kernel='linear', C=0.025, random_state=44),
    sklearn.svm.SVC(gamma=2, C=1, random_state=44),
    sklearn.neighbors.KNeighborsClassifier(3),
    sklearn.naive_bayes.GaussianNB(),
    sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000, random_state=44)
]

# Building a linearly separable dataset (Can be modified)

nb_dimensions = 10

X, y = sklearn.datasets.make_classification(
    n_features=nb_dimensions, n_redundant=0, n_informative=nb_dimensions, random_state=1, n_clusters_per_class=1
)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

linearly_separable = (X, y)

# Definition of some datasets (May need to be modified)

nb_datasets = 100

datasets = [
    #sklearn.datasets.make_moons(noise=0.3, random_state=i)
    sklearn.datasets.make_circles(noise=0.2, factor=0.5, random_state=i)
    for i in range(nb_datasets)
]

# We want to compute the average scores and times of execution

scores = [[] for _ in range(len(classifiers))]
times = [[] for _ in range(len(classifiers))]

# Iteration over datasets

for ds_count, ds in enumerate(datasets):
    # print(f"====== Dataset #{ds_count + 1} ======")

    # We preprocess our dataset and we split it into training and test part

    X, y = ds
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=44
    )

    # Iteration over classifiers

    for clf_count, clf in enumerate(classifiers):
        # print(f"--- {name} ---")

        start_time = time()

        clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        end_time = time()

        scores[clf_count].append(score)
        times[clf_count].append(end_time - start_time)

        # print(f"Obtained score: {score}")
        # print(f"Execution time: {end_time - start_time}")

# Results

z = norm.ppf(0.975)  # 95% CI z-value

for i in range(len(classifiers)):
    print(f"===== {names[i]} =====")

    scores_mean = np.mean(scores[i])
    scores_std_err = np.std(scores[i], ddof=1) / np.sqrt(nb_datasets)
    scores_ci = (scores_mean - z * scores_std_err, scores_mean + z * scores_std_err)

    print(f"Average score: {scores_mean}")
    print(f"--> Confidence interval: {scores_ci}")

    times_mean = np.mean(times[i])
    times_std_err = np.std(times[i], ddof=1) / np.sqrt(nb_datasets)
    times_ci = (times_mean - z * times_std_err, times_mean + z * times_std_err)

    print(f"Average execution time: {times_mean}")
    print(f"--> Confidence interval: {times_ci}")