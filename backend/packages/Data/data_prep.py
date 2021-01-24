import numpy as np
import pandas as pd
import itertools
from datetime import datetime, date
from .data import *

class DataPrep(Data):
	"""
	Store data and makes preparation process.

	:param cols: dictionary with keys {
			'ind': list of column names which together create a unique key and equals 'lev' + 'date',
			'lev': list of column names with level of data,
			'y': list of column names with target values,
			'x': list of column names with data,
			'x_num': list of column names with numerical data,
			'x_cat': list of column names with categorical data}.
	"""

	def __init__(self,
	             cols,
	             mode_test=False,
	             verbose=0):
		"""
		Initialize object of class, save arguments to parameters, initialize parameters.

		:param cols:  dictionary with keys {'lev': list of column names with level of data, 'y': list of column names
		with target values, 'x_num': list of column names with numerical data, 'x_cat': list of column names with
		categorical data}.
		:param mode_test: indicator of test mode
		:param verbose: level of printing out information
		"""
		self.cols = cols
		self.pred_period = pred_period
		self.speed_deact = speed_deact
		self.stock_provided = stock_provided
		self.mode_test = mode_test
		self.verbose = verbose
		self.epsilon = 1e-07

	def __convert_format(self, data_df):
		"""
		Convert columns format to target

		:param data_df: source dataframe
		:return: converted dataframe
		"""
		for column in data_df.columns:
			if column == 'date':
				type_temp = 'datetime64[ns]'
				# data_df[column] = data_df[column].astype(type_temp)
				data_df[column] = pd.to_datetime(data_df[column], dayfirst=True)
			elif column == 'sales':
				type_temp = np.float64
				data_df[column] = data_df[column].astype(type_temp)
			else:
				continue
		return data_df

	def __group_data(self, data_df, cols):
		"""
		Grouping data to the minimum required level: sku, date.

		:param data_df: source dataframe
		:param cols: dictionary with keys {'y': list of column names with target values, 'x': list of column names with
		data}.
		:return: grouped dataframe
		"""

		agg_dict = {}

		for col in cols['y']:
			agg_dict[col] = 'sum'

		for col in cols['x']:
			if data_df[col].dtype in [np.float64, np.int64]:
				agg_dict[col] = np.mean
			else:
				agg_dict[col] = 'first'

		data_df = data_df.groupby(cols['lev'] + [data_df['date'].dt.date], sort=False).agg(agg_dict).reset_index()
		return data_df

	def __split_fill_ind(self, data_df, stock_provided):
		"""
		Split datasets, and fill absent index values

		:param data_df: source dataframe
		:param stock_provided: indicator, that stock (or the same zero sales) are provided.
		:return: train part dataframe, test part dataframe, pred part dataframe
		"""

		if self.mode_test:
			train_df = data_df[(data_df['date'] >= self.date_train_from) &
			                   (data_df['date'] <= self.date_train_to)].copy()
			test_df = data_df[(data_df['date'] >= self.date_pred_from) &
			                  (data_df['date'] <= self.date_pred_to)].copy()
		else:
			train_df = data_df.copy()
			test_df = data_df.drop(data_df.index)

		date_pred_df = pd.DataFrame(index=pd.Index([i.strftime('%Y-%m-%d') for i in pd.date_range(self.date_pred_from, self.date_pred_to)], name='date'))
		lev_train_df = train_df[self.cols['lev']].drop_duplicates()

		if self.mode_test:
			pred_df = data_df[(data_df['date'] >= self.date_pred_from) &
			                  (data_df['date'] <= self.date_pred_to)][self.cols['ind']].copy()
		else:
			pred_df = pd.merge(lev_train_df.assign(key=1), date_pred_df.reset_index().assign(key=1), on=['key']).drop(['key'], axis=1)

		how = 'inner' if self.mode_test else 'right'

		if not stock_provided:
			gb_train_df = train_df.groupby( self.cols['lev'], sort=False )

			# Creating the necessary combinations sku-date
			# ind_train_df = gb_train_df.apply(lambda x: pd.DataFrame(index=pd.Index([i.strftime('%Y-%m-%d') for i in pd.date_range(min(x['date']), max(x['date']))], name='date'))).reset_index()
			ind_train_df = gb_train_df.apply(
				lambda x: pd.DataFrame( index=pd.Index( [i.strftime( '%Y-%m-%d' ) for i in pd.date_range( min( x['date'] ), self.date_train_to )], name='date' ) ) ).reset_index()
			ind_pred_df = gb_train_df.apply( lambda x: date_pred_df ).reset_index()
			ind_test_df = ind_pred_df

			# Filtering by sku-date (Removing unnecessary SKUs, adding empty dates)
			train_df = pd.merge( train_df, ind_train_df, on=self.cols['ind'], how=how )
			pred_df = pd.merge( pred_df, ind_pred_df, on=self.cols['ind'], how=how )
			if self.mode_test:
				test_df = pd.merge( test_df, ind_test_df, on=self.cols['ind'], how=how )
		else:
			# Creating the necessary combinations sku
			lev_pred_df = lev_train_df
			lev_test_df = lev_train_df

			# Filtering by sku (Removing unnecessary sku)
			train_df = pd.merge( train_df, lev_train_df, on=self.cols['lev'], how=how )
			pred_df = pd.merge( pred_df, lev_pred_df, on=self.cols['lev'], how=how )
			if self.mode_test:
				test_df = pd.merge( test_df, lev_test_df, on=self.cols['lev'], how=how )

		return train_df, test_df, pred_df

	def __fill_y(self, train_df, test_df, stock_provided):
		"""
		Fill absent y values if stock not provided

		:param train_df: source train dataframe
		:param test_df: source test dataframe
		:param stock_provided: indicator, that stock (or the same zero sales) are provided.
		:return: filled with zero values train dataframe, test dataframe
		"""
		if not stock_provided:
			train_df[self.cols['y']] = train_df[self.cols['y']].fillna(0)
			test_df[self.cols['y']] = test_df[self.cols['y']].fillna(0)

			# Adjustment if sales are recent (less than 28 days ago)
			#train_df[self.cols['y'][0]] = train_df[self.cols['lev']+self.cols['y']].groupby(self.cols['lev'], sort=False).transform(lambda x: x*min(len(x)/28, 1))

		return train_df, test_df

	def get_prop(self, train_df):
		"""
		Get properties of train dataframe at every level and at total.

		:param train_df: source train dataframe
		:return: properties at every level, properties at total.
		"""

		prop_lev_df = train_df.groupby(self.cols['lev'], sort=False).apply(lambda x: pd.Series({'date_count': max(x['date'].count(), 0),
		                                                                                        'sales_mean': max(x['sales'].mean(), 0),
		                                                                                        'sales_max': max(x['sales'].max(), 0),
		                                                                                        'sales_std': max(x['sales'].std(ddof=0),
		                                                                                                         x['sales'].mean() ** 0.5),  # + self.epsilon
		                                                                                        })).reset_index()

		prop_total = (lambda x: pd.Series({'sales_min': x['sales'].min(),
		                                   'sales_max': x['sales'].max(),
		                                   'sku_count_uniq': x['sku'].nunique(),
		                                   }))(train_df)

		return prop_lev_df, prop_total

	def __prepare_dates(self, data_df):
		"""
		Get key dates for prediction.

		:param data_df: source dataframe
		:return: tuple of key dates
		"""
		if self.mode_test:
			date_cur = (datetime.strptime(data_df['date'].max(), '%Y-%m-%d') - pd.DateOffset(days=self.pred_period - 1)).strftime('%Y-%m-%d')
		else:
			date_cur = date.today().strftime('%Y-%m-%d')

		date_train_from = min(data_df['date'])
		date_train_to = (datetime.strptime(date_cur, '%Y-%m-%d') - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
		date_pred_from = date_cur
		date_pred_to = (datetime.strptime(date_cur, '%Y-%m-%d') + pd.DateOffset(days=self.pred_period - 1)).strftime('%Y-%m-%d')

		if date_train_from > date_train_to:
			raise ValueError('date_train_from > date_train_to')

		if self.verbose >= 1:
			print('self.date_cur', date_cur)
			print('self.date_train_from', date_train_from)
			print('self.date_train_to', date_train_to)
			print('self.date_pred_from', date_pred_from)
			print('self.date_pred_to', date_pred_to)
		return date_cur, date_train_from, date_train_to, date_pred_from, date_pred_to

	def __fit_on_train(self, train_df, date_cur, cols, prop_lev_df, speed_deact):
		"""
		Fit scaler, encoder, filler and others on train data. This function is not allowed to change the training dataset in any way

		:param train_df: source train dataframe
		:param date_cur: current date
		:param cols: dictionary with keys {'ind', 'lev', 'y', 'x', 'x_num', 'x_cat'}.
		:param prop_lev_df: dataframe with properties at every level
		:return: Fit scaler, encoder, filler, corrected columns and others
		"""

		train_df['delta'] = train_df['date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(date_cur, '%Y-%m-%d')).days)
		train_df['weight'] = speed_deact ** np.abs(train_df['delta'])

		# Creating a temporary model factory for seasonality
		cols_seas = cols.copy()
		cols_seas['x'] = []

		models_season = {}

		for index, prop_lev in prop_lev_df.iterrows():
			models_season[prop_lev[cols['lev'][0]]] = SeasonalityFourier(period=[7, 30, 365],
			                                                             fourier_order=[3, 5, 10])

		# Fit models
		train_df.groupby([cols['ind'][0]], sort=False).apply(lambda x: models_season[x.name].fit(x['delta'], x[cols['y'][0]]))
		train_df['season'] = train_df.groupby([cols['ind'][0]], sort=False, group_keys=False, squeeze=True).apply(lambda x: pd.Series(models_season[x.name].transform(x['delta']), x.index))

		# add columns
		# cols['x'] += ['delta', 'season']
		# cols['x_num'] += ['delta', 'season']
		# cols['aux'] += ['weight']

		# del empty columns
		cols_to_del = []
		for col in cols['x']:
			if train_df[col].isnull().values.all():
				cols_to_del += [col]
		cols['x'] = [x for x in cols['x'] if x not in cols_to_del]
		cols['x_num'] = [x for x in cols['x_num'] if x not in cols_to_del]
		cols['x_cat'] = [x for x in cols['x_cat'] if x not in cols_to_del]

		# fillna
		# Be careful, because FillNAWeighted saving indexes and columns of train dataset as pointer
		fillna_d = {}
		fillna_d['lev'] = train_df.groupby(cols['lev'], sort=False).apply(lambda x: FillNAWeighted(cols).fit(train_df, x[cols['x']], x[['delta']]))
		fillna_d['total'] = pd.concat([train_df[cols['x_num']].mean(), train_df[cols['x_cat']].mode().loc[0]])

		# fix values
		train_df[cols['y'][0]] = np.maximum(train_df[cols['y'][0]], 0)  # , where=train_df[cols['y'][0]].values != None

		if 'price' in cols['x']:
			train_df['price'] = np.maximum(train_df['price'], 0)  # , where=train_df['price'].values != None

		# add columns
		cols['x'] += ['delta', 'season']
		cols['x_num'] += ['delta', 'season']
		cols['aux'] += ['weight']

		# scale
		scaler = Scaler(cols)
		scaler.fit(train_df[cols['x_num']])

		# Convert categorical columns to numeric
		mean_encoder = MeanEncoder(self.cols)
		mean_encoder.fit(self.train_df)

		return cols, models_season, mean_encoder, fillna_d, scaler

	def __prep_train(self, train_df, date_cur, cols, fillna_d, scaler, models_season, mean_encoder, speed_deact):
		"""
		Prepare/transform train dataset according to fitted tools

		:param train_df: source train dataset
		:param date_cur: current date
		:param cols: dictionary of columns with keys {'ind', 'lev', 'y', 'x', 'x_num', 'x_cat'}.
		:param fillna_d: dictionary of fillers with keys {'lev', 'total'}
		:param scaler: scaler tool
		:param models_season: seasonal tool
		:param mean_encoder: encoder tool
		:param speed_deact: the speed at which data becomes irrelevant for every day
		:return: prepared train dataset
		"""
		train_df['delta'] = train_df['date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(date_cur, '%Y-%m-%d')).days)
		train_df['weight'] = speed_deact ** np.abs(train_df['delta'])
		train_df['season'] = train_df.groupby([cols['ind'][0]], sort=False, group_keys=False, squeeze=True).apply(lambda x: pd.Series(models_season[x.name].transform(x['delta']), x.index))

		# add missing columns
		for col in cols['x']:
			if col not in train_df.columns:
				train_df[col] = None

		# fillna
		# Be careful, because FillNAWeighted saving indexes and columns of train dataset as pointer
		train_df = train_df.groupby(cols['lev'], sort=False).apply(lambda x: x.fillna((fillna_d['lev'].loc[x.name].transform(x[['delta']])))).reset_index(drop=True)
		train_df = train_df.groupby(cols['lev'], sort=False).apply(lambda x: x.fillna(fillna_d['total'])).reset_index(drop=True)

		# fix values
		train_df[cols['y'][0]] = np.maximum(train_df[cols['y'][0]], 0)  # , where=train_df[cols['y'][0]].values != None
		if 'price' in cols['x']:
			train_df['price'] = np.maximum(train_df['price'], 0)  # , where=train_df['price'].values != None

		# scale
		train_df = scaler.transform(train_df)

		# Convert categorical columns to numeric
		train_df = mean_encoder.transform(train_df)

		train_df = train_df[cols['ind'] + cols['y'] + cols['x'] + cols['aux']]
		return train_df

	def __prep_pred(self, pred_df, date_cur, cols, fillna_d, scaler, models_season, mean_encoder, speed_deact):
		"""
		Prepare/transform predict dataset according to fitted tools.

		:param pred_df: source prediction dataset
		:param date_train_from: train start date
		:param cols: dictionary of columns with keys {'ind', 'lev', 'y', 'x', 'x_num', 'x_cat'}.
		:param fillna_d: dictionary of fillers with keys {'lev', 'total'}
		:param scaler: scaler tool
		:param models_season: seasonal tool
		:param mean_encoder: encoder tool
		:param speed_deact: the speed at which data becomes irrelevant for every day
		:return: prepared predict dataset
		"""

		pred_df['delta'] = pred_df['date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(date_cur, '%Y-%m-%d')).days)
		pred_df['weight'] = speed_deact ** np.abs(pred_df['delta'])
		pred_df['season'] = pred_df.groupby([cols['ind'][0]], sort=False, group_keys=False, squeeze=True).apply(lambda x: pd.Series(models_season[x.name].transform(x['delta']), x.index))

		# add missing columns
		for col in cols['x']:
			if col not in pred_df.columns:
				pred_df[col] = None

		# fillna
		# Be careful, because FillNAWeighted saving indexes and columns of train dataset as pointer
		pred_df = pred_df.groupby(cols['lev'], sort=False).apply(lambda x: x.fillna(fillna_d['lev'].loc[x.name].transform(x[['delta']]))).reset_index(drop=True)
		pred_df = pred_df.groupby(cols['lev'], sort=False).apply(lambda x: x.fillna(fillna_d['total'])).reset_index(drop=True)

		# fix values
		# pred_df[cols['y'][0]] = np.maximum(pred_df[cols['y'][0]], 0) #, where=pred_df[cols['y'][0]].values != None
		if 'price' in cols['x']:
			pred_df['price'] = np.maximum(pred_df['price'], 0)  # , where=pred_df['price'].values != None

		# scale
		pred_df = scaler.transform(pred_df)

		# Convert categorical columns to numeric
		pred_df = mean_encoder.transform(pred_df)

		pred_df = pred_df[cols['ind'] + cols['x'] + cols['aux']]
		return pred_df

	def __prep_test(self, test_df, date_cur, cols, fillna_d, scaler, models_season, mean_encoder, speed_deact):
		"""
		Prepare/transform test dataset according to fitted tools.

		:param test_df:  source test dataset
		:param date_train_from: train start date
		:param cols: dictionary of columns with keys {'ind', 'lev', 'y', 'x', 'x_num', 'x_cat'}.
		:param fillna_d: dictionary of fillers with keys {'lev', 'total'}
		:param scaler: scaler tool
		:param models_season: seasonal tool
		:param mean_encoder: encoder tool
		:param speed_deact: the speed at which data becomes irrelevant for every day
		:return: prepared test dataset
		"""
		test_df['delta'] = test_df['date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(date_cur, '%Y-%m-%d')).days)
		test_df['weight'] = speed_deact ** np.abs(test_df['delta'])
		test_df['season'] = test_df.groupby([cols['ind'][0]], sort=False, group_keys=False, squeeze=True).apply(lambda x: pd.Series(models_season[x.name].transform(x['delta']), x.index))

		# add missing columns
		for col in cols['x']:
			if col not in test_df.columns:
				test_df[col] = None

		# fillna
		# Be careful, because FillNAWeighted saving indexes and columns of train dataset as pointer
		test_df = test_df.groupby(cols['lev'], sort=False).apply(lambda x: x.fillna(fillna_d['lev'].loc[x.name].transform(x[['delta']]))).reset_index(drop=True)
		test_df = test_df.groupby(cols['lev'], sort=False).apply(lambda x: x.fillna(fillna_d['total'])).reset_index(drop=True)

		# fix values
		test_df[cols['y'][0]] = np.maximum(test_df[cols['y'][0]], 0)  # , where=test_df[cols['y'][0]].values != None
		if 'price' in cols['x']:
			test_df['price'] = np.maximum(test_df['price'], 0)  # , where=test_df['price'].values != None

		# scale
		test_df = scaler.transform(test_df)

		# Convert categorical columns to numeric
		test_df = mean_encoder.transform(test_df)

		test_df = test_df[cols['ind'] + cols['y'] + cols['x'] + cols['aux']]
		return test_df

	def prepare(self, data_df):
		"""
		Run prepare process.

		:param data_df:
		:return: returns an instance of self.
		"""
		self.data_df = data_df

		# convert format of columns
		self.data_df = self.__convert_format(self.data_df)

		# group data to target level
		self.data_df = self.__group_data(self.data_df, self.cols)

		# TODO Redo working with dates from strings to the datetime class
		self.data_df['date'] = self.data_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

		# Prepare dates
		(
			self.date_cur,
			self.date_train_from,
			self.date_train_to,
			self.date_pred_from,
			self.date_pred_to
		) = self.__prepare_dates(self.data_df)

		# Split datasets, and fill absent index values
		(
			self.train_df,
			self.test_df,
			self.pred_df
		) = self.__split_fill_ind(self.data_df, self.stock_provided)

		# Fill absent y values
		(
			self.train_df,
			self.test_df,
		) = self.__fill_y(self.train_df,
		                  self.test_df,
		                  self.stock_provided)

		# self.train_df[self.cols['y'][0]] = np.maximum(self.train_df[self.cols['y'][0]].values, 0, where=self.train_df[self.cols['y'][0]].values != None)

		# Collect properties of dataset
		(
			self.prop_lev_df,
			self.prop_total
		) = self.get_prop(self.train_df)

		(
			self.cols,
			self.models_season,
			self.mean_encoder,
			self.fillna_d,
			self.scaler
		) = self.__fit_on_train(self.train_df,
		                        self.date_cur,
		                        self.cols,
		                        self.prop_lev_df,
		                        self.speed_deact)

		self.train_df = self.__prep_train(self.train_df,
		                                  self.date_cur,
		                                  self.cols,
		                                  self.fillna_d,
		                                  self.scaler,
		                                  self.models_season,
		                                  self.mean_encoder,
		                                  self.speed_deact)

		self.pred_df = self.__prep_pred(self.pred_df,
		                                self.date_cur,
		                                self.cols,
		                                self.fillna_d,
		                                self.scaler,
		                                self.models_season,
		                                self.mean_encoder,
		                                self.speed_deact)

		if self.mode_test:
			self.test_df = self.__prep_test(self.test_df,
			                                self.date_cur,
			                                self.cols,
			                                self.fillna_d,
			                                self.scaler,
			                                self.models_season,
			                                self.mean_encoder,
			                                self.speed_deact)

		return self
