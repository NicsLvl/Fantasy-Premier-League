import pandas as pd
import pathlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import numpy as np


gw_data = pathlib.Path.cwd() / 'data' / '2023-24' / 'gws' / 'merged_gw.csv'
df = pd.read_csv(gw_data)

features = ['position_encoded',
            # home away factors
            'was_home', 'diff_ratio',
            # historical point factors
            'avg_ppm', 'sd_ppm', 'avg_ppg', 'sd_ppg',
            # historical goal factors
            'avg_goals', 'sd_goals',
            # historical assist factors
            'avg_assists', 'sd_assists',
            # historical clean sheet factors
            'avg_cs', 'sd_cs',
            # historical goal conceded factors
            'avg_gc', 'sd_gc',
            # historical penalty save factors
            'avg_ps', 'sd_ps',
            # historical penalty miss factors
            'avg_pm', 'sd_pm',
            # historical save factors
            'avg_saves', 'sd_saves',
            # ict index factors
            'influence', 'creativity', 'threat',
            # forecasted factors
            'expected_assists', 'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded',
            # prediction
            'total_points']

# create a difficulty ratio
fixture_data = pathlib.Path.cwd() / 'data' / '2023-24' / 'fixtures.csv'
fix_df = pd.read_csv(fixture_data)
df = pd.merge(df, fix_df[fix_df['finished']][['id', 'team_h_difficulty', 'team_a_difficulty']], how='left', left_on='fixture', right_on='id')
df['diff_ratio'] = df.apply(lambda x: x['team_h_difficulty']/x['team_a_difficulty'] if x['was_home'] else x['team_a_difficulty']/x['team_h_difficulty'], axis=1)

# create points per minute
df['ppm'] = df['total_points'] / df['minutes']
df['ppm'].fillna(0, inplace=True)

# start calculating mean and std dev of previous games
df = df.sort_values(by=['name', 'GW'])
df['avg_ppm'] = df.groupby('name')['ppm'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_ppm'] = df.groupby('name')['ppm'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_ppg'] = df.groupby('name')['total_points'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_ppg'] = df.groupby('name')['total_points'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_cs'] = df.groupby('name')['clean_sheets'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_cs'] = df.groupby('name')['clean_sheets'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_goals'] = df.groupby('name')['goals_scored'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_goals'] = df.groupby('name')['goals_scored'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_assists'] = df.groupby('name')['assists'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_assists'] = df.groupby('name')['assists'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_gc'] = df.groupby('name')['goals_conceded'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_gc'] = df.groupby('name')['goals_conceded'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_ps'] = df.groupby('name')['penalties_saved'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_ps'] = df.groupby('name')['penalties_saved'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_pm'] = df.groupby('name')['penalties_missed'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_pm'] = df.groupby('name')['penalties_missed'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)
df['avg_saves'] = df.groupby('name')['saves'].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df['sd_saves'] = df.groupby('name')['saves'].apply(lambda x: x.shift().expanding().std()).reset_index(level=0, drop=True)

le = LabelEncoder()
df['position_encoded'] = le.fit_transform(df['position'])
df = df[features]
df = df.dropna()

pos = df['position_encoded'].unique()
corrs = {}
for p in pos:
    print(f'Position: {le.inverse_transform([p])[0]}')
    print(df[df['position_encoded'] == p].corr()['total_points'].sort_values(ascending=False))
    corrs[p] = df[df['position_encoded'] == p].corr()['total_points'].sort_values(ascending=False).index[1:5]

print(corrs)
models = {}

print('Random Forest Regressor')
for p in pos:
    print(f'Position: {le.inverse_transform([p])[0]}')
    X = df[df['position_encoded'] == p].drop(columns=['total_points'])
    X = X[corrs[p]]
    y = df[df['position_encoded'] == p]['total_points']
    print(f'Average Total Points: {y.mean()}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    models[le.inverse_transform([p])[0]] = model

    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'R2: {model.score(X_test, y_test)}')
    print(f'Feature Names: {X.columns}')
    importance_str = ', '.join(f'{importance:.2f}' for importance in model.feature_importances_)
    print(f'Feature Importance: {importance_str}')
    print('')

print('Gradient Boosting Regressor Grid Search')
for p in pos:
    print(f'Position: {le.inverse_transform([p])[0]}')
    X = df[df['position_encoded'] == p].drop(columns=['total_points'])
    X = X[corrs[p]]
    y = df[df['position_encoded'] == p]['total_points']
    print(f'Average Total Points: {y.mean()}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = GradientBoostingRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3],
        'learning_rate': [0.01, 0.02, 0.03]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Best Score: {grid_search.best_score_}')
    y_pred = grid_search.predict(X_test)
    models[le.inverse_transform([p])[0]] = grid_search

    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'R2: {r2_score(y_test, y_pred)}')
    print(f'Feature Names: {X.columns}')
    importance_str = ', '.join(f'{importance:.2f}' for importance in grid_search.best_estimator_.feature_importances_)
    print(f'Feature Importance: {importance_str}')
    print('')

print(models)

# calculate points for current week players

# use algo to determine best starting 11

# use algo to determine points for current team and then calculate the best transfer
