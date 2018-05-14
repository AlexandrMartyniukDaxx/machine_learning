from diabetes_indians.lib_4 import load_diabetes_df, norm_df, plot_class


df_init = load_diabetes_df()
df_norm = norm_df(df_init)
print(df_norm.head(10))

plot_class(df_init, df_norm)
