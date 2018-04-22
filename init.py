import pandas as pd
import numpy as np

titanic_df = pd.read_csv('titanic.csv')

# Passengers total
total_rows = titanic_df.shape[0]
print('Passengers total:', total_rows, '\n')

titanic_df['AgeGroup'] = pd.cut(titanic_df.Age, [0, 18, 45, 65, 90])
class_set = titanic_df.groupby(['Pclass', 'Sex', 'AgeGroup', 'Survived'])['PassengerId'].count()
grouped_table = class_set.groupby(level=['Pclass']).apply(lambda x: x / float(x.sum()) * 100)

print(class_set)
print(grouped_table)


# Pivot table sex/class
pvt_sex = titanic_df.pivot_table(index=['Sex'], columns=['Pclass'], values='Name', aggfunc='count')
print('Pivot on Sex/Class:', '\n', pvt_sex, '\n')

# Pivot table survived/class
df_survived = titanic_df[titanic_df.Survived == 1];
total_survived = df_survived.shape[0]
pvt_survived = df_survived.pivot_table(index=['Pclass', 'Sex'], values=['Name'], aggfunc='count')
pvt_survived['%'] = pvt_survived.apply(lambda row: row.values / total_survived * 100)
# pvt_survived['Tot%'] = pvt_survived.apply(lambda row: row.values / total_rows * 100)
print('Total survived: ', total_survived)
print('Pivot on Survived/Class/Sex:', '\n', pvt_survived, '\n')

