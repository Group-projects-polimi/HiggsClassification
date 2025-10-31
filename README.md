## Machine Learning Project 1

Authors: Francesco Derme, Pietro Fumagalli, Saransh Chopra

This repository contains our project 1 submission for the Machine Learning (CS-433) course at EPFL. The code addresses the challenge of predicting coronary heart disease (MICHD) from the BRFSS health dataset. We propose and analyze two types of models which correspond to two different ways of doing classification: logistic regression and random forests. We show that these models achieve good F1-scores on the testing data, thus providing an effective method for risk assessment. Finally, we explore an ensemble model, combining both the previous models and giving recommendations to guide future research on such medical datasets.

## Repository structure

The repository is structures in the following way:

```
.
├── README.md                 # this file (high level overview of the work)
├── data.py                   # data pre-processing pipeline
├── dataset                   # directory storing the dataset files
│   ├── x_test.csv            # testing features (git ignored)
│   ├── x_train.csv           # training features (git ignored)
│   └── y_train.csv           # training labels (git ignored)
├── dataset_processed         # directory created by run.py to store pre-processed dataset
│   ├── x_test_clean.csv      # pre-processed testing features (git ignored)
│   └── x_train_clean.csv     # pre-processed training featured (git ignored)
├── logistic_regression.py    # our enhanced logistic regression model
├── implementations.py        # functions required for the assignment
├── metrics.py                # shared metrics used to evaluate the models
├── plotting_util.py          # plotting functions for the report
├── random_forest.py          # our random forest model
├── requirements.txt          # exact dependencies of our code
└── run.py                    # script to run and reproduce our work
```

Furthermore, each module is documented with docstrings and comments.

## Data pipeline

The entire data pre-processing pipeline is written in `data.py` module. The module does not contain functions required at the time of training, such as the function for generating a minibatch. These functions are defined in the respective model module.

The data pipeline performs the following operations on the dataset:
1. Load data from CSV files. (`load_csv`)
2. Manual preprocessing to fix known data issues.
3. Split features into numerical and categorical based on predefined indices.
4. Remove features with >60% NaN values.
5. Remove highly correlated features, keeping the one with fewest NaNs.
6. Impute and fill missing values.
7. Standardize numerical features to have zero mean and unit variance.
8. Selects categorical features based on association, predictiveness, and NaN count
9. One-hot encode categorical features.
10. Save processed features and labels to CSV files.

Please take a look at individual functions' docstrings for further information.

## Models and Methods

The models and methods are distributed in `implementations.py`, `logistic_regression.py`, and `random_forest.py`. `implementations.py` contains functions required in the assignment and, `logistic_regression.py` and `random_forest`.py contain our custom implementations. We do not include our ensamble model (consisting of both the previously mentioned models) as the model did not perform very well and the idea is not mature enough.

1. **Logistic regression**: A linear model whose output is converted to a probability by the logistic function. The model is trained using the cross-entropy loss which is smooth and better suited for the task than the F1 evaluation metric. The loss is weighted to account for class imbalance and a plethora of options based on the relative frequencies of the two classes (about 10:1 in the preprocessed dataset) were tested. The loss' gradient is regularized to combat overfitting and the hyperparameter $\lambda$, the regularization strength, is chosen via stratified cross-validation on 5 folds. The training uses the \textitAdam optimizer combined with an exponential-decay learning-rate scheduler. These two techniques, the optimizer and the scheduler, are independent and complementary: the first manages updates of specific weights based on the global rate provided by the second. The training algorithm has a maximum number of epochs (these are "passes" over the whole dataset) which is never reached because training stops if there are no validation-loss improvements in a certain number of epochs (patience). Cross validation not only returns the best $\lambda$, but also the ideal number of epochs $\alpha$ to train for. We then perform a final training over $\alpha$ epochs without carving a validation set from the dataset. This does not risk overfitting as we don't give the model enough time to learn the noise in the data. As a threshold is needed to turn probabilities into predictions, we choose the best by evaluating candidates using F1 scores during cross validation. Candidates are chosen by taking quantiles of the predicted probabilities, so that more thresholds are tested in the regions where the model predicts most probabilitie. After careful hyperparameter tuning and several ablation tests, the model achieved 0.429 F1 score on the test set.

2. **Random Forest**: A random forest is a powerful ensemble method, ideal for non-linear relationships and heterogeneous data. Each decision tree is composed of nodes and each node splits the data by finding the feature and threshold that minimize the Gini impurity, which measures how good a given node is at separating the data. The tree grows recursively until a stopping criterion (like max depth or min samples split) is met, at which point leaf nodes learn to predict the majority class of the samples they were reached by. At inference time, each sample takes a unique path in any given tree and it's predicted to be part of the class of the leaf it reaches. A random forest is an ensemble of many such decision trees, in particular we used 200. We leveraged two key techniques to reduce variance and prevent overfitting: bagging, which means each tree is trained on a different bootstrap sample (a random sample with replacement) of the training data, and feature randomness, which means that at each split the tree is only allowed to consider a random subset of the total features. We conducted ablation studies focusing on three key components: data sampling, tree regularization, and prediction thresholding. Our experiments demonstrate that balanced bootstrap sampling with a 2:1 ratio of negative to positive class ensures that each tree learns the patterns of the rare positive class. Furthermore, our experiments show that relatively shallow trees (15-20 levels) can produce a better F1 score than deeper trees (25-30 levels) by being less prone to overfitting. We implemented a dedicated threshold tuning function to search a wide range of threshold candidates. Our best random forest achieves its peak F1 score at a threshold of 0.2, which means that 20% of the trees need to vote positive for a subject to be considered at-risk. After careful hyperparameter tuning and several ablation tests, the model achieved 0.425 F1 score on the test set.

Please take a look at individual functions' docstrings for further information.

## Usage

> [!NOTE]
> Training the logistic regression model requires ~1 hour and pre-processing th dataset requires ~20 minutes. It might take more time depending on your system's hardware configuration. You can use the random forest model for faster training, but it will not perform as good as the logistic regression model.

To run the code:

1. Create and activate a Python virtual environment using your favourite tool
2. Install requirements.txt
3. Run run.py, which will reproduce our results by:
   1. Pre-processing the data,
   2. Training logistic regression (and optionally random forest - uncomment/comment lines starting with RANDOM FOREST in the file), and
   3. Generating predictions
