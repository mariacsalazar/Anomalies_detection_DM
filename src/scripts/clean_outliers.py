import numpy as np
import pandas as pd
import datetime
import read_write as io
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def quantil_dataframe(df):
    '''
    Filter period and percentiles
    '''
    df['period_m'] = pd.to_datetime(df['timestamps_UTC']).dt.strftime('%Y%m')
    df = df[df['period_m'] >= '2023-04-01']
    cols_use = set(df.columns) - {'Unnamed: 0',
                                  'mapped_veh_id',
                                  'timestamps_UTC',
                                  'lat',
                                  'lon',
                                  'period'}
    dfQuantil = df[list(cols_use)].describe(percentiles=[0.05, .25, .75, 0.9])
    return dfQuantil


def df_treshold(df):
    '''
    Allows to have the upper whisker of a boxplot
    '''
    q5 = df.loc[['5%']].values.flatten().tolist()
    q25 = df.loc[['25%']].values.flatten().tolist()
    q75 = df.loc[['75%']].values.flatten().tolist()
    q90 = df.loc[['90%']].values.flatten().tolist()
    lim_inf = [w - 1.5 * (y - x) for w, x, y, z in zip(q5, q25, q75, q90)]
    lim_sup = [z + 1.5 * (y - x) for w, x, y, z in zip(q5, q25, q75, q90)]
    df.loc['lim_inf'] = lim_inf
    df.loc['lim_sup'] = lim_sup
    limit_airTemp_up = df.loc[['lim_sup']].RS_E_InAirTemp_PC1.values[0]
    limit_rpm_up = df.loc[['lim_sup']].RS_E_RPM_PC1.values[0]
    limit_waterT_up = df.loc[['lim_sup']].RS_E_WatTemp_PC1.values[0]
    limit_oilT_up = df.loc[['lim_sup']].RS_T_OilTemp_PC1.values[0]
    limit_oilP_up = df.loc[['lim_sup']].RS_E_OilPress_PC1.values[0]

    limit_airTemp_down = df.loc[['lim_inf']].RS_E_InAirTemp_PC1.values[0]
    limit_rpm_down = df.loc[['lim_inf']].RS_E_RPM_PC1.values[0]
    limit_waterT_down = df.loc[['lim_inf']].RS_E_WatTemp_PC1.values[0]
    limit_oilT_down = df.loc[['lim_inf']].RS_T_OilTemp_PC1.values[0]
    limit_oilP_down = df.loc[['lim_inf']].RS_E_OilPress_PC1.values[0]
    return limit_airTemp_up, limit_rpm_up, limit_waterT_up, limit_oilT_up, limit_oilP_up, \
        limit_airTemp_down, limit_rpm_down, limit_waterT_down, limit_oilT_down, limit_oilP_down


def change_outliers(df,
                    limit_airTemp_up,
                    limit_rpm_up,
                    limit_waterT_up,
                    limit_oilT_up,
                    limit_oilP_up,
                    limit_airTemp_down,
                    limit_rpm_down,
                    limit_waterT_down,
                    limit_oilT_down,
                    limit_oilP_down
                    ):
    '''change the outliers for the median'''
    df['period_m'] = pd.to_datetime(
        df['timestamps_UTC']).dt.strftime('%Y%m').astype(int)
    dfMedian = df[df['period_m'] >= 202304]
    df['RS_E_InAirTemp_PC1'] = np.where(
        df['RS_E_InAirTemp_PC1'].between(limit_airTemp_down, limit_airTemp_up),
        df['RS_E_InAirTemp_PC1'], np.nan)
    df['RS_E_InAirTemp_PC2'] = np.where(
        df['RS_E_InAirTemp_PC2'].between(limit_airTemp_down, limit_airTemp_up),
        df['RS_E_InAirTemp_PC2'], np.nan)
    df['RS_E_RPM_PC1'] = np.where(
        df['RS_E_RPM_PC1'].between(limit_rpm_down, limit_rpm_up),
        df['RS_E_RPM_PC1'], np.nan)
    df['RS_E_RPM_PC2'] = np.where(
        df['RS_E_RPM_PC2'].between(limit_rpm_down, limit_rpm_up),
        df['RS_E_RPM_PC2'], np.nan)
    df['RS_E_WatTemp_PC1'] = np.where(
        df['RS_E_WatTemp_PC1'].between(limit_waterT_down, limit_waterT_up),
        df['RS_E_WatTemp_PC1'], np.nan)
    df['RS_E_WatTemp_PC2'] = np.where(
        df['RS_E_WatTemp_PC2'].between(limit_waterT_down, limit_waterT_up),
        df['RS_E_WatTemp_PC2'], np.nan)
    df['RS_T_OilTemp_PC1'] = np.where(
        df['RS_T_OilTemp_PC1'].between(limit_oilT_down, limit_oilT_up),
        df['RS_T_OilTemp_PC1'], np.nan)
    df['RS_T_OilTemp_PC2'] = np.where(
        df['RS_T_OilTemp_PC2'].between(limit_oilT_down, limit_oilT_up),
        df['RS_T_OilTemp_PC2'], np.nan)
    df['RS_E_OilPress_PC1'] = np.where(
        df['RS_E_OilPress_PC1'].between(limit_oilP_down, limit_oilP_up),
        df['RS_E_OilPress_PC1'], np.nan)
    df['RS_E_OilPress_PC2'] = np.where(
        df['RS_E_OilPress_PC2'].between(limit_oilP_down, limit_oilP_up),
        df['RS_E_OilPress_PC2'], np.nan)
    return df


def imp_transform(df):
    imp_mean = IterativeImputer(random_state=42, max_iter=10)
    df_train = df.loc[:, ['RS_E_InAirTemp_PC1',
                          'RS_E_InAirTemp_PC2',
                          'RS_E_OilPress_PC1',
                          'RS_E_OilPress_PC2',
                          'RS_E_RPM_PC1',
                          'RS_E_RPM_PC2',
                          'RS_E_WatTemp_PC1',
                          'RS_E_WatTemp_PC2',
                          'RS_T_OilTemp_PC1',
                          'RS_T_OilTemp_PC2']]
    imp_mean.fit(df_train)
    df_imp = imp_mean.transform(df_train)
    df.loc[:, ['RS_E_InAirTemp_PC1',
               'RS_E_InAirTemp_PC2',
               'RS_E_OilPress_PC1',
               'RS_E_OilPress_PC2',
               'RS_E_RPM_PC1',
               'RS_E_RPM_PC2',
               'RS_E_WatTemp_PC1',
               'RS_E_WatTemp_PC2',
               'RS_T_OilTemp_PC1',
               'RS_T_OilTemp_PC2']] = df_imp
    return df


def main():
    df = io.read_data(io.Filenames.data_cleaned)
    dfQuantil = quantil_dataframe(df)
    limit_airTemp_up, limit_rpm_up, limit_waterT_up, limit_oilT_up, limit_oilP_up, \
        limit_airTemp_down, limit_rpm_down, limit_waterT_down, limit_oilT_down, limit_oilP_down = df_treshold(
            dfQuantil)
    df = change_outliers(df,
                         limit_airTemp_up,
                         limit_rpm_up,
                         limit_waterT_up,
                         limit_oilT_up,
                         limit_oilP_up,
                         limit_airTemp_down,
                         limit_rpm_down,
                         limit_waterT_down,
                         limit_oilT_down,
                         limit_oilP_down)
    df = imp_transform(df)
    df.drop(['period_m'], axis=1, inplace=True)
    io.write_data(df, io.Filenames.outliers_fixed)
    return


if __name__ == "__main__":
    main()
