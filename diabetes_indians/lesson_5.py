import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from diabetes_indians.lib_4 import load_diabetes_df, norm_df, norm_minmax
from diabetes_indians.lib_5 import predict, stratified_split, calc_accuracy, calc_bcr_accuracy


def complex_validation(df):
    nfold = 100
    key = 'class'
    columns = ['U_LR', 'U_LR_BCR', 'STD_LR', 'STD_LR_BCR', 'MM_LR', 'MM_LR_BCR',
                                   'U_RF', 'U_RF_BCR', 'STD_RF', 'STD_RF_BCR', 'MM_RF', 'MM_RF_BCR']

    rows = []
    for i in range(nfold):
        train, test = stratified_split(df[key])

        x_train = df.iloc[train, 0:8]
        x_test = df.iloc[test, 0:8]
        x_train_std = norm_df(x_train)
        x_test_std = norm_df(x_test)
        x_train_mm = norm_minmax(x_train)
        x_test_mm = norm_minmax(x_test)
        y_train = df[key][train]
        y_test = df[key][test]

        logreg = LogisticRegression()
        rf = RandomForestClassifier()

        u_logreg_pred = predict(x_train, y_train, x_test, y_test, logreg)
        std_logreg_pred = predict(x_train_std, y_train, x_test_std, y_test, logreg)
        mm_logreg_pred = predict(x_train_mm, y_train, x_test_mm, y_test, logreg)

        u_rf_pred = predict(x_train, y_train, x_test, y_test, rf)
        std_rf_pred = predict(x_train_std, y_train, x_test, y_test, rf)
        mm_rf_pred = predict(x_train_mm, y_train, x_test, y_test, rf)

        rows.append([
            calc_accuracy(y_test, u_logreg_pred),
            calc_bcr_accuracy(y_test, u_logreg_pred),
            calc_accuracy(y_test, std_logreg_pred),
            calc_bcr_accuracy(y_test, std_logreg_pred),
            calc_accuracy(y_test, mm_logreg_pred),
            calc_bcr_accuracy(y_test, mm_logreg_pred),
            calc_accuracy(y_test, u_rf_pred),
            calc_bcr_accuracy(y_test, u_rf_pred),
            calc_accuracy(y_test, std_rf_pred),
            calc_bcr_accuracy(y_test, std_rf_pred),
            calc_accuracy(y_test, mm_rf_pred),
            calc_bcr_accuracy(y_test, mm_rf_pred)])

    return pd.DataFrame(data=rows, columns=columns).mean()


df = load_diabetes_df()

# print('LogReg normalized', cross_validate(df, 'class', LogisticRegression(), 100, True))
# print('LogReg unnormalized', cross_validate(df, 'class', LogisticRegression(), 100, False))
# print('Forest normalized', cross_validate(df, 'class', RandomForestClassifier(), 100, True))
# print('Forest unnormalized', cross_validate(df, 'class', RandomForestClassifier(), 100, False))

print(complex_validation(df))
