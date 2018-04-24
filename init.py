import pandas as pd
import nameResolver as nr

titanic_df = pd.read_csv('titanic.csv')

# Passengers total
total_rows = titanic_df.shape[0]
print('Passengers total:', total_rows)
survivers_set = titanic_df.groupby(['Survived'])['PassengerId'].count()
print('\nSurvivors set:')
print(survivers_set)

titanic_df['AgeGroup'] = pd.cut(titanic_df.Age, [0, 18, 45, 65, 90])
class_set = titanic_df[titanic_df['Survived'] == 1].groupby(['Pclass', 'Sex', 'AgeGroup'])['PassengerId'].count()
grouped_table = class_set.groupby(level=['Pclass']).apply(lambda x: x / float(x.sum()) * 100)

print('\nPercentage of survivors divided by sex and age per class')
print(grouped_table)

# Pivot table sex/class
pvt_sex = titanic_df.pivot_table(index=['Sex'], columns=['Pclass'], values='PassengerId', aggfunc='count')
print('\nPivot on Sex/Class:', '\n', pvt_sex, '\n')

# Pivot table survived/class
titanic_df['ShortName'] = titanic_df['Name'].apply(lambda x: nr.get_name(x))
df_survived = titanic_df[titanic_df.Survived == 1]

total_survived = df_survived.shape[0]
pvt_survived = df_survived.pivot_table(index=['Pclass', 'Sex'], values=['PassengerId'], aggfunc='count')
pvt_survived['%'] = pvt_survived.apply(lambda row: row.values / total_survived * 100)

print('Pivot table on Survived by Class, Sex')
print(pvt_survived)


gpd_survived_females = df_survived[df_survived['Sex'] == 'female'].groupby(['Pclass', 'ShortName'])['ShortName'].count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False) \
                             .head(3)
print('\nSurvived females:')
print(gpd_survived_females)


gpd_survived_males = df_survived[df_survived['Sex'] == 'male'].groupby(['Pclass', 'ShortName'])['ShortName'].count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False) \
                             .head(3)
print('\nSurvived males:')
print(gpd_survived_males)
