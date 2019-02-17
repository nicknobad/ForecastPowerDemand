# At the moment the script works with Initial Demand Outturn. To train on the Final Demand Outturn data you have to download it
# Steps:
# 0. Inputs
# 1. FUN for holidays, doy, moy, dow
# 2. Configs
# 3. Download and prepare (clean, process and format) the datasets
#    2A. Demand
#    2B. Weather
# 4. Train, CV and Test
#    4A. Simple CV
#    4B. Predictions on train, val and test
#    4C. ToDo: Plot best model preds in the test dataset vs real
#    4D. ToDo: Tune several models and benchmark their performance
# 5. Download weather forecast
# 6. Download demand forecast from NG; Combine 6A with 6B
#    6A. DA HHly demand Forecast from NG
#    6B. OC2-14 Peak demand. Explode it to HHly.
# 7. Merge Fct Demand and Weather (from Steps 5 and 6) for Step 8. Get all time vars and Holidays
#    Forecast HHly demand for the next 2 weeks and plot it
                                                ################
                                                #### Step 0 ####
                                                ################
import pandas as pd, configparser, numpy as np
from datetime import datetime as ddt; import datetime as dt
print('{}: START'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:') ))
# To solve warning: "A value is trying to be set on a copy of a slice from a DataFrame."; Disable chained assignments
pd.options.mode.chained_assignment = None
# because Elexon has a 2000-row limit on the downloadable files, the script must loop over months: 31 days * 48 = 1488 rows
date_start, date_end = '2017-01-01', ddt.now().strftime('%Y-%m-%d')
# create from_ and to_date columns covering today
df_days = pd.DataFrame( { 'date': pd.date_range(date_start, date_end, freq='d') })
df_days['from_date'] = df_days['date'].values.astype('datetime64[M]')
df_days['to_date'] = df_days['from_date'] + pd.tseries.offsets.MonthEnd(1)
df_months = df_days[['from_date', 'to_date']].drop_duplicates().reset_index(drop=True)

# df to map hour and period
# ?? Daylight saving time ?? create date and period df and then merge with df_weather_Hist_avg
# df_HistDmd_zz = df_HistDmd.loc[df_HistDmd['SettlementPeriod'].isin([49, 50])] # last Sunday in Oct?
# timechange in Oct and Mar: period 49 and 50 should match 23:00 and 23:30 (?); 02:00 and 02:30 will match their incremental periods
df_map_HH_Period = pd.DataFrame({ 'time': pd.date_range('00:00', '23:30', freq='30min').time, 'SettlementPeriod': range(1, 49) })
                                                ################
                                                #### Step 1 ####
                                                ################
print('{}: Reading config file'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:') ))
# get holidays in a df with their names (return_name = True) and IDs
from pandas.tseries.holiday import ( AbstractHolidayCalendar, DateOffset, EasterMonday, GoodFriday, Holiday, MO, next_monday, next_monday_or_tuesday)
class Cal_EngWales(AbstractHolidayCalendar):
    rules = [ GoodFriday, EasterMonday,
        Holiday('New Years Day', month=1, day=1, observance=next_monday),
        Holiday('Early May bank holiday', month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Spring bank holiday', month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer bank holiday', month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas Day', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day', month=12, day=26, observance=next_monday_or_tuesday)
        ]
    # assign id to each hol; non-hol will have id_hol = 0
    df_gen_hols = pd.DataFrame({ 'name_hol': [hol.name for hol in rules], 'id_hol': range(1, len(rules)+1) })
    # get: name_hol; id_hol (assign '0' to non-hols); dow, doy, moy
    def get_hols(self, DF, col_date):
        series_hols = self.holidays(start = DF[col_date].min(), end = DF[col_date].max(), return_name = True)
        df_hols = pd.DataFrame({ col_date: pd.DataFrame( series_hols.index )[0].dt.date, 'name_hol': series_hols.tolist() })
        # merge to get name_ and id_hol; assign '0' to non-holidays
        DF[col_date] = pd.to_datetime(DF[col_date]).dt.date        
        DF = pd.merge( pd.merge( DF, df_hols[[col_date, 'name_hol']], how='left', on=col_date ), self.df_gen_hols, how='left', on='name_hol' )        
        DF['id_hol'].fillna(0, inplace=True)
        DF['dow'], DF['doy'], DF['moy'] = pd.to_datetime(DF[col_date]).dt.dayofweek, pd.to_datetime(DF[col_date]).dt.dayofyear, pd.to_datetime(DF[col_date]).dt.month
        return DF
                                                ################
                                                #### Step 2 ####
                                                ################
try: # read configs
    config = configparser.RawConfigParser()
    config.read_file(open(r'config_FctDmd.cfg'))
except Exception as e:
    print( 'ERROR: {}'.format(str(e)) )
    exit()
# API Elexon key, version; OC2-14 Peak Demand forecast
key_API_elexon, version = config.get('API_elexon', 'key_api_elexon'), config.get('API_elexon', 'version')
API_Fct2_14PkDmd = 'https://api.bmreports.com/BMRS/DEMMF2T14D/{}?APIKey={}&ServiceType=csv'.format( version, key_API_elexon )
# weather: stations, from_ and to_dates - get previous day too
# Weather History
list_stations = config.get('API_weather_history', 'list_stations').split(',')
date_yest = (ddt.strptime(date_start, '%Y-%m-%d') - dt.timedelta(days=1)).strftime( '%Y-%m-%d' )
year1, month1, day1 = ddt.strptime(date_yest, '%Y-%m-%d').year, ddt.strptime(date_yest, '%Y-%m-%d').month, ddt.strptime(date_yest, '%Y-%m-%d').day
year2, month2, day2 = ddt.strptime(date_end, '%Y-%m-%d').year, ddt.strptime(date_end, '%Y-%m-%d').month, ddt.strptime(date_end, '%Y-%m-%d').day
# Weather Forecast; https://github.com/weatherbit/weatherbit-python; https://www.weatherbit.io/forecast/16
list_cities, key_API_WeatherFct = config.get('API_weather_forecast', 'list_cities').split(','), config.get('API_weather_forecast', 'key_API')
                                                ################
                                                #### Step 3 ####
                                                ################
                                                #### Step 3A ###
# Download dmd data month by month to avoid too large API calls (>2k rows)
df_HistDmd = pd.DataFrame()
for index, row in df_months.iterrows():
    date_from, date_to = str(row['from_date'].date()), str(row['to_date'].date())
    API_HistDmd = 'https://api.bmreports.com/BMRS/INDOITSDO/{}?APIKey={}&FromDate={}&ToDate={}&ServiceType=csv'.format( version, key_API_elexon, date_from, date_to )
    # download data, convert the multiIndex to columns, rename ALL columns
    print('{}: Downloading indo from {} to {}'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), date_from, date_to ))
    df_tmp = pd.read_csv( API_HistDmd, sep = ',' )
    df_tmp.reset_index(inplace=True)
    df_tmp.columns = ['RecordType', 'SettlementDate', 'SettlementPeriod', 'SystemZone', 'date_publ', 'dmd']
    # drop rows that contain: 'ITSDO', 'FTR'
    df_tmp = df_tmp[~df_tmp['RecordType'].isin( ['ITSDO', 'FTR'] )]
    df_HistDmd = df_HistDmd.append( df_tmp )

# convert SettlementDate to date, reset_index
df_HistDmd['SettlementDate'] = pd.to_datetime(df_HistDmd['SettlementDate'], format='%Y%m%d').dt.date
df_HistDmd = df_HistDmd.reset_index(drop=True)
# find the peak demand per day and HHly coeffs
df_HistDmd['dmd_pk'] = df_HistDmd.groupby('SettlementDate')['dmd'].transform('max')
df_HistDmd['dmd_coeff'] = df_HistDmd['dmd'] / df_HistDmd['dmd_pk']

# get only periods for HHs: merge df_HistDmd with df_map_HH_Period; drop nan rows - rows with period = 49 and 50
df_HistDmd = pd.merge( df_HistDmd, df_map_HH_Period, how = 'left', on = ['SettlementPeriod'] )
                                            #### Step 3B ####
### To avoid creating too large API calls - download each city and append the DFs
df_weather_Hist = pd.DataFrame(); print()
for idcity in list_stations:
    print('{}: Downloading weather for {}'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), idcity ))
    API_weather = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={}&data=tmpc&year1={}&month1={}&day1={}&year2={}&month2={}&day2={}&tz=Etc%2FUTC&format=onlycomma&latlon=yes&direct=yes&report_type=1&report_type=2'.format( idcity, year1, month1, day1, year2, month2, day2 )
    df_tmp_weather = pd.read_csv( API_weather, sep = ',', header = 0 )
    df_weather_Hist = df_weather_Hist.append(df_tmp_weather)
# format columns; clean temp column from any non-numeric data
df_weather_Hist['valid'] = pd.to_datetime(df_weather_Hist['valid'])
df_weather_Hist['tmpc'] = pd.to_numeric(df_weather_Hist[ 'tmpc' ], errors='coerce')
df_weather_Hist.dropna(inplace=True)

# simple daily average; more accurate would be to have a population- or other-weighted average temperature; other measure for wind speed
df_weather_Hist_avg = df_weather_Hist[['valid', 'tmpc']].groupby('valid').mean().reset_index()
df_weather_Hist_avg['datetime'] = df_weather_Hist_avg['valid'].dt.round('30min')
del df_weather_Hist_avg['valid']
df_weather_Hist_avg['date'], df_weather_Hist_avg['time'] = df_weather_Hist_avg['datetime'].dt.date, df_weather_Hist_avg['datetime'].dt.time

# merge DFs: Dmd, Weather; fitler out today; get hols, doy, moy, dow
df_merged_Hist_ALL = pd.merge( df_HistDmd, df_weather_Hist_avg, how = 'left', left_on = ['SettlementDate', 'time'], right_on = ['date', 'time'] )
df_merged_Hist_ALL = df_merged_Hist_ALL.loc[df_merged_Hist_ALL['SettlementDate'].astype(str) != date_end]
df_merged_Hist_ALL = Cal_EngWales().get_hols(DF = df_merged_Hist_ALL, col_date = 'SettlementDate')

# MISSING WEATHER DATA FOR SOME HALF-HOURS - THAT'S WHY len(df_merged_Hist_ALL) != df_merged_Hist
# df_merged_Hist_zz = df_merged_Hist_ALL[df_merged_Hist_ALL['tmpc'].isnull()]
# df_weather_Hist_zz = df_weather_Hist.loc[ (df_weather_Hist['valid'] >= '2017-02-13 17:30:00') & (df_weather_Hist['valid'] <= '2017-02-13 22:00:00') ]
                                                ################
                                                #### Step 4 ####
                                                ################
# list of desired columns
list_cols = ['SettlementDate', 'SettlementPeriod', 'id_hol', 'dow', 'doy', 'moy', 'dmd_pk', 'tmpc', 'dmd_coeff']
df_merged_Hist = df_merged_Hist_ALL[ list_cols ]
# drop nulls
df_merged_Hist.dropna(inplace = True)
# get correlation coefficients(pearsons coeffs)
df_merged_Hist.corr()['dmd_coeff'].sort_values(ascending=False)
#                        # Some EDA
#                        # scatterplot
#                        df_merged_Hist.plot(x='SettlementPeriod', y='dmd_pk', kind='scatter')
#                        
#                        # plotting pairs
#                        import matplotlib.pyplot as plt
#                        fig = pd.plotting.scatter_matrix(df_merged_Hist, alpha=0.2, figsize = (70, 50), diagonal='hist') # 'kde'
#                        plt.tight_layout()
#                        plt.savefig('fig1.png')
#                        
#                        # df summary/describe
#                        zz1 = df_merged_Hist.describe()
#                        zz1.reset_index(level=0, inplace=True)
#                        # wide to long
#                        zz2 = pd.melt(zz1, id_vars = ['index'], value_vars = [xx for xx in list(zz1.columns) if xx != 'index'])
#                        # long to wide
#                        zz2 = zz2.pivot(index = 'variable', columns = 'index', values = 'value')
#                        # /Some EDA

# split to train, val and test
ratio_train, ratio_test_val = 0.6, 0.4 # their sum must equal to 1
df_train, df_val, df_test = np.array_split(df_merged_Hist.sample(frac=1), [int( (ratio_train)*len(df_merged_Hist)), int( (2*ratio_test_val)*len(df_merged_Hist))])
df_train.name, df_val.name, df_test.name = 'train', 'val', 'test'
list_cols_train = ['SettlementPeriod', 'id_hol', 'dow', 'doy', 'moy', 'dmd_pk', 'tmpc']
col_Y = 'dmd_coeff'
                                            #### Step 4A ####
# PyMC3: https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Project.ipynb
# Models: https://scikit-learn.org/stable/modules/classes.html
# linear models: https://scikit-learn.org/stable/modules/linear_model.html
# Theory: https://scikit-learn.org/stable/supervised_learning.html
# train and benchmark models
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLarsCV, ElasticNet, RANSACRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
# laptop is shutting down
# from sklearn.kernel_ridge import KernelRidge

# prepare models
list_models = []
list_models.append(('LR', LinearRegression() ))
# list_models.append(('BR', BayesianRidge() ))
# list_models.append(('LassoLarsCV', LassoLarsCV() ))
# list_models.append(('LinearSVR', LinearSVR() ))
# list_models.append(('SVR-rbf', SVR(kernel='rbf') ))
# list_models.append(('SVR-rbf', SVR(kernel='linear') ))
# list_models.append(('SVR-rbf', SVR(kernel='poly') ))
# list_models.append(('KNeighborsRegr', KNeighborsRegressor(n_neighbors=3) ))
list_models.append(('GradBoostRegr', GradientBoostingRegressor() ))
list_models.append(('RandForestRegr', RandomForestRegressor() ))
list_models.append(('AdaBoostRegr', AdaBoostRegressor() ))
list_models.append(('BaggingRegr', BaggingRegressor() ))
list_models.append(('ExtraTreesRegr', ExtraTreesRegressor() ))
# list_models.append(('ElasticNet', ElasticNet(max_iter=1000) ))
# list_models.append(('RANSACRegr', RANSACRegressor() ))

# Cross-validate models; save CV metrics results in a df: Mean Absolute Error (MAE), Mean Squared Error (MSE), R2
# Available metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
dict_scoring = { 'MAE': 'neg_mean_absolute_error', 'MSE': 'neg_mean_squared_error', 'R2': 'r2', 'Explained Var': 'explained_variance',
                 'MSLogE': 'neg_mean_squared_log_error', 'MedAE': 'neg_median_absolute_error'}
df_cv_metrics_Long = pd.DataFrame()
for name, model in list_models:
    for kk, vv in dict_scoring.items():
        kfold = model_selection.KFold(n_splits=10)
        results_cv = model_selection.cross_val_score(model, df_train[list_cols_train], df_train[col_Y].values, cv = kfold, scoring = vv)
        print( '{}: {}, {}: Mean {:f} (std {:f})'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), name, kk, results_cv.mean(), results_cv.std()) )
        _df_cv_metrics = pd.DataFrame({ 'model': name, 'metric': kk, 'value': results_cv })
        df_cv_metrics_Long = df_cv_metrics_Long.append( _df_cv_metrics ).reset_index(drop=True)
print( '\n{}: Finished with CVing'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:') ))
# convert long to wide and boxplot it
df_boxplot_cv = df_cv_metrics_Long.loc[df_cv_metrics_Long['metric'] == 'R2']
str_metric = list(df_boxplot_cv['metric'])[0]
str_title_boxplot = 'CV. Model Benchmark by {}'.format( str_metric )
df_boxplot_cv.boxplot(column = 'value', by = 'model')
plt.title( str_title_boxplot )
plt.suptitle('')
plt.show()
# average the metrics and convert long to wide
df_cv_metrics_Wide_avg = df_cv_metrics_Long.groupby( ['model', 'metric'] ).mean().reset_index()
df_cv_metrics_Wide_avg = df_cv_metrics_Wide_avg.pivot(index = 'model', columns = 'metric', values = 'value')
df_cv_metrics_Wide_avg.reset_index(level=0, inplace=True)
                                            #### Step 4B ####
# fit chosen models. Option: automate the process of choosing which models to fit
def TraindPreds(lst_models, lst_DFs, DF_train, DF_val, DF_test, cols_train, col_Y, lst_metric_cols):
    df_metr = pd.DataFrame()
    for _model in lst_models:
        # fit the models on train
        print( '\n{}: Fitting {}'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), _model[0] ))
        model_fit = _model[1].fit( DF_train[cols_train], DF_train[col_Y].values )
        # predicting and calculating errors on traind, val and test
        for _df in lst_DFs:
            print( '{}: Predicting with {} on {}'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), _model[0], _df.name ))
            _df[ 'pred.' + _model[0] ] = model_fit.predict( _df[list_cols_train] )
            print( '{}: Calculating errors with {} on {}'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), _model[0], _df.name ))
            _df['Err.' + _model[0]] = _df[col_Y] - _df['pred.' + _model[0]]
            _df['abs%Err.' + _model[0]] = abs(_df['Err.' + _model[0]]) / _df[col_Y]
            # metrics
            print( '{}: Saving metrics '.format( ddt.now().strftime('%d%b%Y %H:%M:%S:') ))
            _df_metr = pd.DataFrame([[ _df.name, _model[0], _df['abs%Err.' + _model[0]].mean(),
                                       metrics.r2_score( _df[col_Y], _df[ 'pred.' + _model[0] ]),
                                       metrics.mean_absolute_error( _df[col_Y], _df[ 'pred.' + _model[0] ]),
                                       metrics.mean_squared_error( _df[col_Y], _df[ 'pred.' + _model[0] ]),
                                       metrics.mean_squared_log_error( _df[col_Y], _df[ 'pred.' + _model[0] ]),
                                       metrics.median_absolute_error( _df[col_Y], _df[ 'pred.' + _model[0] ])]],
                        columns = lst_metric_cols
                        )
            df_metr = pd.concat([df_metr, _df_metr]).reset_index(drop=True)
    return df_metr
# define list of DFs to predict and calculate errors on; prepare metric DFs
list_DFs = [ df_train, df_val, df_test ]
list_metrics = [ 'df', 'model', 'MAPE', 'R2', 'MAE', 'MSE', 'MSLogE', 'MedAE' ]
# run the Fun
df_metrics = TraindPreds( lst_models = list_models, lst_DFs = list_DFs, DF_train = df_train, DF_val = df_val, DF_test = df_test,
             cols_train = list_cols_train, col_Y = col_Y, lst_metric_cols = list_metrics )
df_metrics['combo'] = df_metrics['R2'] * (1-df_metrics['MAPE'])

df_metric_train = df_metrics.loc[df_metrics['df'] == 'train']
df_metric_val = df_metrics.loc[df_metrics['df'] == 'val']
df_metric_test = df_metrics.loc[df_metrics['df'] == 'test']
# choose the best model according to a metric
model_best_cv = df_cv_metrics_Wide_avg.loc[df_cv_metrics_Wide_avg['R2'].idxmax()][0]
model_best_val = df_metric_val.loc[df_metric_val['combo'].idxmax()][1]
model_best_test = df_metric_test.loc[df_metric_test['combo'].idxmax()][1]

if model_best_cv in [model_best_val, model_best_test]:
    model_chosen = model_best_cv
else:
    print('''\n--Something's up!! {} not in {}'''.format( model_best_cv, [model_best_val, model_best_test] ))
    print('--Best model in CV not best in val and in test datasets/n')
# plot metrics
# df_metrics.pivot(index='df', columns='model', values='MSE').plot()
# df_metrics.dtypes

# get datetime column from date and period by mapping time and period with "df_map_HH_Period"
# plt.scatter(x = df_train['pred.GradBoostRegr'], y = df_train['dmd_coeff'])

###################################################################################################################################################
                                                ################
                                                #### Step 5 ####
                                                ################
from weatherbit.api import Api # pip install pyweatherbit
api_call = Api(key_API_WeatherFct)
api_call.set_granularity('daily') # 'hourly', '3hourly'
df_weather_Fct = pd.DataFrame()
for iCity in list_cities:
    print( '{}: Downloading Weather Forecast for {}'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:'), iCity ))
    tmp_df_weather_fct = pd.DataFrame( api_call.get_forecast(city = iCity, country = 'GB').get_series(['temp', 'max_temp', 'min_temp', 'precip']) )
    tmp_df_weather_fct['City'] = iCity
    df_weather_Fct = df_weather_Fct.append( tmp_df_weather_fct )
# simple daily average; more accurate would be to have a population- or other-weighted average temperature; other measure for wind speed
df_weather_Fct_avg = df_weather_Fct[['datetime', 'temp']].groupby('datetime').mean().reset_index()
df_weather_Fct_avg['date'], df_weather_Fct_avg['time'] = df_weather_Fct_avg['datetime'].dt.date, df_weather_Fct_avg['datetime'].dt.time
df_weather_Fct_avg.rename(columns={'temp':'tmpc'}, inplace=True)
                                                ################
                                                #### Step 6 ####
                                                ################
                                                #### Step 6A ###
# DA Forecast Demand from NG; get dmd_coeff and dmd_pk; rename columns
print( '{}: Downloading DA Peak Demand and processing'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:') ))
date_tmrw = (ddt.now() + dt.timedelta(days=1)).strftime( '%Y-%m-%d' )
API_DmdDA = 'https://api.bmreports.com/BMRS/B0620/{}?APIKey={}&SettlementDate={}&Period={}&ServiceType=csv'.format( version, key_API_elexon, date_tmrw, '*' )
df_FctDmdDA = pd.read_csv( API_DmdDA, skiprows = 4, usecols = [1, 2, 3] )
df_FctDmdDA.dropna(inplace=True)
df_FctDmdDA['Settlement Date'] = pd.to_datetime(df_FctDmdDA['Settlement Date']).dt.date
df_FctDmdDA['type_fct'], df_FctDmdDA['dmd_pk'], df_FctDmdDA['DA_dmd_coeff'] = 'fct_NG', df_FctDmdDA['Quantity'].max(), df_FctDmdDA['Quantity']/df_FctDmdDA['Quantity'].max()
df_FctDmdDA_Temps = pd.merge(df_FctDmdDA, df_weather_Fct_avg[['date', 'tmpc']], how='left', left_on='Settlement Date', right_on='date')
df_FctDmdDA_Temps.rename(columns={'Settlement Date':'sett_date', 'Settlement Period':'SettlementPeriod', 'Quantity':'DA_Dmd'}, inplace=True)
                                            #### Step 6B ####
# download OC2-14 peak demand forecast; convert columns; subset the DF; explode to HHly with cross join
print( '{}: Downloading OC2-14 Peak Demand and processing\n'.format( ddt.now().strftime('%d%b%Y %H:%M:%S:') ))
df_PkDmd2_14_Fct = pd.read_table(API_Fct2_14PkDmd, sep=',', skiprows=[0], header=None, names=['type_dmd', 'sett_date', 'SettlementPeriod', 'id_boundary', 'date_publ', 'dmd_pk'])
df_PkDmd2_14_Fct = df_PkDmd2_14_Fct[~df_PkDmd2_14_Fct['type_dmd'].isin( ['FTR'] )]
df_PkDmd2_14_Fct['sett_date'] = pd.to_datetime(df_PkDmd2_14_Fct['sett_date'].astype(str), format='%Y%m%d').dt.date
df_PkDmd2_14_Fct['date_publ'] = pd.to_datetime(df_PkDmd2_14_Fct['date_publ'], format='%Y%m%d%H%M%S.0')
df_PkDmd2_14_Fct = df_PkDmd2_14_Fct.loc[df_PkDmd2_14_Fct['type_dmd'] == 'DSN']
df_PkDmd2_14_Fct_HHly = pd.merge(df_PkDmd2_14_Fct[['sett_date', 'dmd_pk']].assign(key=0), df_map_HH_Period[['SettlementPeriod']].assign(key=0), on='key').drop('key', axis=1)
                                                ################
                                                #### Step 7 ####
                                                ################
# Forecast with the chosen model; re-arrange columns
model_used = [item[1] for item in list_models if model_best_cv == item[0]][0]
name_model_used = [item[0] for item in list_models if model_best_cv == item[0]][0]
# merge Fct weather and df_PkDmd2_14_Fct_HHly; add DA Demand Forecast from NG; get id_hol, dow, doy, moy; predict dmd_coeff and calculate HHly dmd
df_merged_Fct = pd.merge(df_PkDmd2_14_Fct_HHly, df_weather_Fct_avg[['date', 'tmpc']], how='left', left_on=['sett_date'], right_on=['date'])
df_merged_Fct['type_fct'] = 'fct_predicted'
# combine DA Dmd Fct with the predictions
df_merged_Fct = pd.concat([df_merged_Fct[['type_fct', 'sett_date', 'SettlementPeriod', 'tmpc', 'dmd_pk']],
                           df_FctDmdDA_Temps[['type_fct', 'sett_date', 'SettlementPeriod', 'tmpc', 'dmd_pk', 'DA_dmd_coeff', 'DA_Dmd' ]]
                           ])
# get hols, doy, moy, dow; predict on Fct df; sort by date and period
df_merged_Fct = Cal_EngWales().get_hols(DF = df_merged_Fct, col_date = 'sett_date')
df_merged_Fct[ 'dmd_coeff.' + model_best_cv ] = model_used.predict( df_merged_Fct[list_cols_train] )
df_merged_Fct['pred.' + model_best_cv] = df_merged_Fct[ 'dmd_coeff.' + model_best_cv ] * df_merged_Fct[ 'dmd_pk' ]

df_merged_Fct.sort_values(['sett_date', 'SettlementPeriod'], inplace=True)

# plot the results
df_merged_Fct.pivot(index='SettlementPeriod', columns='sett_date', values='pred.' + model_best_cv).plot()
