import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from time import time

import sklearn.datasets
import sklearn.ensemble
import sklearn.metrics
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
    "Linear SVM",
    "RBF SVM",
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

accuracies = [[] for _ in range(len(classifiers))]
precisions = [[] for _ in range(len(classifiers))]
recalls = [[] for _ in range(len(classifiers))]
f1s = [[] for _ in range(len(classifiers))]

times = [[] for _ in range(len(classifiers))]

# Iteration over datasets

for ds_count, ds in enumerate(datasets):
    # We preprocess our dataset and we split it into training and test part

    X, y = ds
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=ds_count
    )

    # Iteration over classifiers

    for clf_count, clf in enumerate(classifiers):
        start_time = time()

        # Fitting the model
        clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), clf)
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)

        end_time = time()

        # Computation of metrics
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        precision = sklearn.metrics.precision_score(y_test, y_pred, average='binary', zero_division=0.0)
        recall = sklearn.metrics.recall_score(y_test, y_pred)
        f1 = sklearn.metrics.f1_score(y_test, y_pred)

        accuracies[clf_count].append(accuracy)
        precisions[clf_count].append(precision)
        recalls[clf_count].append(recall)
        f1s[clf_count].append(f1)

        times[clf_count].append(end_time - start_time)

# Results

z = norm.ppf(0.975)  # 95% CI z-value

for i in range(len(classifiers)):
    print(f"========== {names[i]} ==========")
    print()

    # Accuracy

    a_mean = np.mean(accuracies[i])
    a_std_err = np.std(accuracies[i], ddof=1) / np.sqrt(nb_datasets)
    a_ci = (a_mean - z * a_std_err, a_mean + z * a_std_err)

    print(f"Average accuracy: {a_mean}")
    print(f"Standard deviation of accuracy: {a_std_err}")
    print(f"--> Confidence interval: {a_ci}")

    # Precision

    p_mean = np.mean(precisions[i])
    p_std_err = np.std(precisions[i], ddof=1) / np.sqrt(nb_datasets)
    p_ci = (p_mean - z * p_std_err, p_mean + z * p_std_err)

    print(f"Average precision: {p_mean}")
    print(f"Standard deviation of precision: {p_std_err}")
    print(f"--> Confidence interval: {p_ci}")

    # Recall

    r_mean = np.mean(recalls[i])
    r_std_err = np.std(recalls[i], ddof=1) / np.sqrt(nb_datasets)
    r_ci = (r_mean - z * r_std_err, r_mean + z * r_std_err)

    print(f"Average recall: {r_mean}")
    print(f"Standard deviation of recall: {r_std_err}")
    print(f"--> Confidence interval: {r_ci}")

    # F1-score

    f_mean = np.mean(f1s[i])
    f_std_err = np.std(f1s[i], ddof=1) / np.sqrt(nb_datasets)
    f_ci = (f_mean - z * f_std_err, f_mean + z * f_std_err)

    print(f"Average F1-score: {f_mean}")
    print(f"Standard deviation of F1-score: {f_std_err}")
    print(f"--> Confidence interval: {f_ci}")

    print()

    times_mean = np.mean(times[i])
    times_std_err = np.std(times[i], ddof=1) / np.sqrt(nb_datasets)
    times_ci = (times_mean - z * times_std_err, times_mean + z * times_std_err)

    print(f"Average execution time: {times_mean}")
    print(f"Standard deviation of execution time: {times_std_err}")
    print(f"--> Confidence interval: {times_ci}")

    print()

# Plotting box plots and error bars
# (Accuracies, F1-scores)

plt.figure()

x = range(1, len(names)+1)

plt.subplot(2, 1, 1)
plt.boxplot(accuracies, labels=names,
            boxprops={'color': 'blue'}, medianprops={'color': 'red'},
            flierprops={'marker': '+'}, whiskerprops={'linestyle': 'dotted'})
plt.title('Box Plot of Accuracy with Different Classifiers')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 1, 2)
plt.boxplot(f1s, labels=names,
            boxprops={'color': 'blue'}, medianprops={'color': 'red'},
            flierprops={'marker': '+'}, whiskerprops={'linestyle': 'dotted'})
plt.title('Box Plot of F1-Score with Different Classifiers')
plt.ylabel('F1-score')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()