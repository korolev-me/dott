import numpy as np
import pandas as pd
from io import StringIO
import re

from .data import Data

class DataUpload(Data):
	"""
	Store data and makes uploading process.
	"""
	def __init__(self, userpath):
		"""
		Initialize object of class, save arguments to parameters, initialize parameters.

		:param userpath: path to user folder with source data.
		"""
		self.userpath = userpath

	def readDataFrame(self, file):
		"""
		Read source file and validate that its columns and index are valid.

		:param file: path or stream to source file
		:return: uploaded and validated dataframe
		"""

		read_correct = False

		data_str = str(file.stream.read(), 'utf-8')
		for sep in [',', ';', '\t']:
			try:
				data_df = pd.read_csv(StringIO(data_str), sep=sep)
				if len(data_df.columns) > 1:
					read_correct = True
					break
			except Exception as error:
				print('error: ', error)
				continue

		if read_correct == False:
			raise ValueError('Cant read file') # + repr(error)

		self.data_df = data_df