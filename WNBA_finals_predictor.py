# WNBA Finals Predictor


import pandas as pd



# info on points scored for each team in wnba reg season: https://www.basketball-reference.com/wnba/years/2023_games.html
schedule = pd.read_csv("reg_season.csv")
schedule = schedule.iloc[:, 1:-2]
schedule.head()


# advanced stats taken from: https://www.basketball-reference.com/wnba/years/2023.html
advanced_stats = pd.read_csv("advanced_stats.csv")
advanced_stats = advanced_stats.dropna(axis=1, how='all')
advanced_stats = advanced_stats.iloc[:, 1:-1]
advanced_stats.head()


df = pd.merge(schedule, advanced_stats, left_on="Visitor/Neutral", right_on="Team")
df = pd.merge(df, advanced_stats, left_on="Home/Neutral", right_on="Team")
df = df.drop(['Team_x', 'Team_y'], axis=1)
df.head()



# Add a column for to show if the home team won
for index, row in df.iterrows():
    if df.loc[index, 'PTS'] > df.loc[index, 'PTS.1']:
        # Place 0 for home loss
        df.loc[index, 'Home_Winner'] = 0
    else:
        # Place 1 for home win
        df.loc[index, 'Home_Winner'] = 1

remove_cols = ["Home/Neutral", "Visitor/Neutral", "Home_Winner", "PTS", "PTS.1"]


selected_cols = [x for x in df.columns if x not in remove_cols]


df[selected_cols].head()


# Scale data
from sklearn.preprocessing import MinMaxScaler


scalar = MinMaxScaler()
df[selected_cols] = scalar.fit_transform(df[selected_cols])
df.head()




## Determine predictors


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier


# Initialize ridge regression classification
rr = RidgeClassifier(alpha=1.0)


sfs = SequentialFeatureSelector(rr, n_features_to_select=10, direction='backward')


sfs.fit(df[selected_cols], df['Home_Winner'])


# Create a list of the most impactful columns
predictors = list(df[selected_cols].columns[sfs.get_support()])
df[predictors].head()




## Create and test model


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def monte_carlo(n):
    accuracy = []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['Home_Winner'], test_size=0.2)

        # Train a logistic regression model on the training data
        model = LogisticRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))

    # Get the average accuracy
    score = sum(accuracy) / len(accuracy)
    return score


score = monte_carlo(1000)
print(f"Accuracy: {score}")




## Predict the Finals


# Remove the rows where the Aces and Liberty play against each other
non_aces_liberty_game = df
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')].index)
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'New York Liberty') & (non_aces_liberty_game['Visitor/Neutral'] == 'Las Vegas Aces')].index)
non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')]


# Get a game where the Aces are home and Liberty is away
final_matchup = df[(df['Home/Neutral'] == 'Las Vegas Aces') & (df['Visitor/Neutral'] == 'New York Liberty')][:1]
final_matchup[predictors]


# Predict the winner of the final matchup
model = LogisticRegression()
model.fit(non_aces_liberty_game[predictors], non_aces_liberty_game['Home_Winner'])


# Predict the outcome of the final_matchup. Prediction of 1.0 means the home team will win, and 0.0 means the away team will win
y_pred = model.predict(final_matchup[predictors])
print(f"Prediction: {y_pred[0]}")

