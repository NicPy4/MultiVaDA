import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 0. Setting parameters for run
csv_path = 'air_quality.csv'
num_trials = 15


# 1. Reading the data
data = pd.read_csv(csv_path, sep=';')
print(f"Read data with shape: {data.shape} and columns: {data.columns}.\n")

x = data.drop(['indoor PM1', 'indoor PM2.5', 'outdoor PM10', 'outdoor PM2.5'], axis=1)
y = data[['indoor PM1', 'indoor PM2.5', 'outdoor PM10', 'outdoor PM2.5']]

print(f"x data:\n{x}\n")
print(f"y data:\n{y}\n")

# Visualizing the correlation matrix for the input features
plt.figure(figsize=(22, 22))
x_corr = x.corr().round(2)
mask = np.zeros_like(x_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(x_corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0, cbar=False, annot=True, linewidths=0.5, annot_kws={"size": 16})
plt.title('Correlation matrix of input features', fontsize=40)
plt.savefig('results/correlation_matrix.pdf')
plt.show()


# 2. Set up and train a non-linear support vector regressor
regressor = make_pipeline(StandardScaler(), svm.SVR())
regressor = MultiOutputRegressor(regressor)
regressor.fit(x.values, y.values)
print(f"Prediction: {x.values[1]}\nYields: {regressor.predict(x.values[1].reshape(1, -1))}.\nGround truth: {y.values[1]}\n")

# Plot predicted vs ground truth for indoor PM1 and PM2.5
predicted = regressor.predict(x.values)
xs = np.arange(predicted.shape[0])
plt.plot(xs, predicted[:, 0], label='Predicted indoor PM1')
plt.plot(xs, y.values[:, 0], label='Ground truth indoor PM1')
plt.legend()
plt.savefig('results/simple_predicted_vs_ground_truth.pdf')
plt.show()


# 3. Validation and nested cross-validation of the model
non_nested_scores = np.zeros(num_trials)
nested_scores = np.zeros(num_trials)

scaler = StandardScaler()
scaler.fit(x.values, y=y.values[:, 0])
scaled_x = scaler.transform(x.values)
scaled_y = y.values[:, 0]

p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}

for i in range(num_trials):
    # Choose cross-validation techniques for the inner and outer loops
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non-nested parameter search and scoring
    regressor = GridSearchCV(estimator=svm.SVR(), param_grid=p_grid, cv=outer_cv, n_jobs=-1, scoring="r2")
    regressor.fit(scaled_x, scaled_y)
    non_nested_scores[i] = regressor.best_score_

    # Nested CV with parameter optimization
    regressor = GridSearchCV(estimator=svm.SVR(), param_grid=p_grid, cv=inner_cv, n_jobs=-1, scoring="r2")
    nested_score = cross_val_score(regressor, X=scaled_x, y=scaled_y, cv=outer_cv, n_jobs=-1)
    nested_scores[i] = nested_score.mean()

score_difference = non_nested_scores - nested_scores


# 4. Plot scores on each trial for nested and non-nested CV
print(f"Average difference of {score_difference.mean()} with std. dev. of {score_difference.std()}.\n")

plt.figure()
plt.subplot(211)
(non_nested_scores_line,) = plt.plot(non_nested_scores, color="r")
(nested_line,) = plt.plot(nested_scores, color="b")
plt.ylabel("score", fontsize="14")
plt.legend(
    [non_nested_scores_line, nested_line],
    ["Non-Nested CV", "Nested CV"],
    bbox_to_anchor=(0, 0.4, 0.5, 0),
)
plt.title(
    "Non-Nested and Nested Cross Validation on Iris Dataset",
    x=0.5,
    y=1.1,
    fontsize="15",
)

plt.subplot(212)
difference_plot = plt.bar(range(num_trials), score_difference)
plt.xlabel("Individual Trial #")
plt.ylabel("Score difference", fontsize="14")

plt.savefig("results/cross_validation.pdf")
plt.show()
