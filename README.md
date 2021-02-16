# KNNAnalysis
A short exploration into the KNN algorithm and a brief comparison to linear regression

## 1 K-Nearest Neighbours for Regression
This notebook explores the use of 5 fold cross validation to estimate k and distance metric.

## 2 Speed
Here we examine the KDTree data structure provided by sklearn to see how this can be used to improve run time.

## 3 K-Nearest Neighbours for Classification
This notebook explores the use of KNNs for classification problems. The validation method here is a simple three set split between training, validation and test.

## 4 Vs Linear Model
This notebook explores the effectiveness of linear models against the KNN algorithm. The linear model in this case has its weights set through the use of the closed form solution of the empirical risk minimization problem. This utilizes the economical SVD to calculate the weights.
