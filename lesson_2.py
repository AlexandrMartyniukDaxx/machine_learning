import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 4)), columns=list('ABCD'))
Y = pd.DataFrame(np.random.randint(0, 2, size=(1000, 1)), columns=list('Y'))

print()

# print(new_df)
print(df.median(axis=0))
print(Y)
