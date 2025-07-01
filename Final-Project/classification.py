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

nb_datasets = 200

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
    current_test_size = np.random.uniform(0.2, 0.5) # Generates a random float between 0.2 and 0.5

    X, y = ds
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=current_test_size, random_state=ds_count 
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
decimal_places = 4 # number of decimal places 

for i in range(len(classifiers)):
    print(f"\n========== {names[i]} ==========") 
    print()

    # Accuracy
    a_mean = np.mean(accuracies[i])
    a_std_err = np.std(accuracies[i], ddof=1) / np.sqrt(nb_datasets)
    a_ci_lower = a_mean - z * a_std_err
    a_ci_upper = a_mean + z * a_std_err

    print(f"Average accuracy: {a_mean:.{decimal_places}f}")
    print(f"Standard deviation of accuracy: {a_std_err:.{decimal_places}f}")
    print(f"--> Confidence interval: ({a_ci_lower:.{decimal_places}f}, {a_ci_upper:.{decimal_places}f})")

    # Precision
    p_mean = np.mean(precisions[i])
    p_std_err = np.std(precisions[i], ddof=1) / np.sqrt(nb_datasets)
    p_ci_lower = p_mean - z * p_std_err
    p_ci_upper = p_mean + z * p_std_err

    print(f"Average precision: {p_mean:.{decimal_places}f}")
    print(f"Standard deviation of precision: {p_std_err:.{decimal_places}f}")
    print(f"--> Confidence interval: ({p_ci_lower:.{decimal_places}f}, {p_ci_upper:.{decimal_places}f})")

    # Recall
    r_mean = np.mean(recalls[i])
    r_std_err = np.std(recalls[i], ddof=1) / np.sqrt(nb_datasets)
    r_ci_lower = r_mean - z * r_std_err
    r_ci_upper = r_mean + z * r_std_err

    print(f"Average recall: {r_mean:.{decimal_places}f}")
    print(f"Standard deviation of recall: {r_std_err:.{decimal_places}f}")
    print(f"--> Confidence interval: ({r_ci_lower:.{decimal_places}f}, {r_ci_upper:.{decimal_places}f})")

    # F1-score
    f_mean = np.mean(f1s[i])
    f_std_err = np.std(f1s[i], ddof=1) / np.sqrt(nb_datasets)
    f_ci_lower = f_mean - z * f_std_err
    f_ci_upper = f_mean + z * f_std_err

    print(f"Average F1-score: {f_mean:.{decimal_places}f}")
    print(f"Standard deviation of F1-score: {f_std_err:.{decimal_places}f}")
    print(f"--> Confidence interval: ({f_ci_lower:.{decimal_places}f}, {f_ci_upper:.{decimal_places}f})")

    times_mean = np.mean(times[i])
    times_std_err = np.std(times[i], ddof=1) / np.sqrt(nb_datasets)
    times_ci_lower = times_mean - z * times_std_err
    times_ci_upper = times_mean + z * times_std_err

    print(f"Average execution time: {times_mean:.{decimal_places}f} seconds") # Added 'seconds'
    print(f"Standard deviation of execution time: {times_std_err:.{decimal_places}f} seconds")
    print(f"--> Confidence interval: ({times_ci_lower:.{decimal_places}f}, {times_ci_upper:.{decimal_places}f}) seconds")

    print()

# Convert lists of lists to numpy arrays for easier plotting
accuracies = np.array(accuracies).transpose()
precisions = np.array(precisions).transpose()
recalls = np.array(recalls).transpose()
f1s = np.array(f1s).transpose()
times_np = np.array(times).transpose() # Convert times to numpy array for plotting

# Define a common plotting function for Box Plot and Confidence Interval Error Bar
def plot_metric_distributions(fig_obj, axes_obj, data_metric, metric_name, names_list, nb_datasets, z_val):
    # Set the window title
    fig_obj.canvas.manager.set_window_title(f'{metric_name} Performance Distribution')

    # Box Plot
    axes_obj[0].boxplot(data_metric, tick_labels=names_list, notch=True,
                        boxprops={'color': 'blue'}, medianprops={'color': 'red'},
                        flierprops={'marker': '+'}, whiskerprops={'linestyle': 'dotted'})
    axes_obj[0].set_title(f'Box Plot of {metric_name} Across Replications', fontsize=12)
    axes_obj[0].set_ylabel(metric_name, fontsize=10)
    axes_obj[0].grid(True, linestyle='--', alpha=0.6)
    axes_obj[0].tick_params(axis='x', rotation=90, labelsize=9)
    plt.setp(axes_obj[0].get_xticklabels(), ha="center")

    # Confidence Interval Error Bar (formerly "Error Bar")
    means = np.mean(data_metric, axis=0)
    std_errors = np.std(data_metric, axis=0, ddof=1) / np.sqrt(nb_datasets) # Calculate Standard Error
    ci_margin = z_val * std_errors # Calculate Margin of Error for 95% CI

    axes_obj[1].errorbar(names_list, means, yerr=ci_margin, fmt='o', color='green', capsize=5, ecolor='red')
    axes_obj[1].set_title(f'Mean {metric_name} with 95% Confidence Intervals', fontsize=12)
    axes_obj[1].grid(True, linestyle='--', alpha=0.6)
    axes_obj[1].tick_params(axis='x', rotation=90, labelsize=9)
    plt.setp(axes_obj[1].get_xticklabels(), ha="center")

    fig_obj.suptitle(f'{metric_name} Performance Distribution and CI', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])


# Function for Bar Charts with 95% CI for overall comparison
def plot_bar_chart_with_ci(data_metric, metric_name, names_list, nb_datasets, z_val):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f'{metric_name} Overall Comparison (Bar Chart)')

    means = np.mean(data_metric, axis=0)
    std_errors = np.std(data_metric, axis=0, ddof=1) / np.sqrt(nb_datasets)
    ci_margin = z_val * std_errors

    x_pos = np.arange(len(names_list))
    
    ax.bar(x_pos, means, yerr=ci_margin, align='center', alpha=0.7, color='skyblue', capsize=10, ecolor='black')
    
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_list, rotation=45, ha="right", fontsize=9)
    ax.set_title(f'Overall {metric_name} Comparison with 95% Confidence Intervals', fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    
    # Add mean values on top of bars
    for i, mean_val in enumerate(means):
        ax.text(x_pos[i], mean_val + ci_margin[i] + 0.01, f'{mean_val:.{decimal_places}f}', 
                ha='center', va='bottom', fontsize=8, color='darkblue')

    plt.tight_layout()

# Plotting Calls

# Box plots and Confidence Interval Error Bars (for all metrics)
fig_acc, axes_acc = plt.subplots(1, 2, figsize=(18, 7))
plot_metric_distributions(fig_acc, axes_acc, accuracies, 'Accuracy', names, nb_datasets, z)

fig_prec, axes_prec = plt.subplots(1, 2, figsize=(18, 7))
plot_metric_distributions(fig_prec, axes_prec, precisions, 'Precision', names, nb_datasets, z)

fig_rec, axes_rec = plt.subplots(1, 2, figsize=(18, 7))
plot_metric_distributions(fig_rec, axes_rec, recalls, 'Recall', names, nb_datasets, z)

fig_f1, axes_f1 = plt.subplots(1, 2, figsize=(18, 7))
plot_metric_distributions(fig_f1, axes_f1, f1s, 'F1-score', names, nb_datasets, z)

# Distribution for times if relevant
fig_time, axes_time = plt.subplots(1, 2, figsize=(18, 7))
plot_metric_distributions(fig_time, axes_time, times_np, 'Execution Time', names, nb_datasets, z)

# Bar Charts for Overall Comparison (with 95% CI)
plot_bar_chart_with_ci(accuracies, 'Accuracy', names, nb_datasets, z)
plot_bar_chart_with_ci(precisions, 'Precision', names, nb_datasets, z)
plot_bar_chart_with_ci(recalls, 'Recall', names, nb_datasets, z)
plot_bar_chart_with_ci(f1s, 'F1-score', names, nb_datasets, z)
plot_bar_chart_with_ci(times_np, 'Execution Time', names, nb_datasets, z)

plt.show()