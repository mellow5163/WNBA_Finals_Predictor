# WNBA Finals Predictor

This program uses logistic regression to predict the winner of WNBA games based on data from the 2023 WNBA regular season. This model is built using Python, pandas, and sci-kit learn.

### Model Overview

- A **Ridge Regression Classifier** is used to select the top 10 most impactful features from the dataset.

- The model's accuracy is evaluated using a **Monte Carlo simulation**, which trains and tests the model 1,000 times on different random splits of the data.





### Data Sources

- **reg_season.csv:** contains game results from the 2023 WNBA regular season.

- **advanced_stats.csv:** contains additional statistics for each team that participated in the 2023 WNBA regular season.


### Usage & Output

To make a prediction, run the script using Python. The script outputs a prediction of either 1.0 (home team wins) or 0.0 (away team wins), as well as the average accuracy of the model.
