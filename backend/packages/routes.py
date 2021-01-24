import os
import time
from datetime import datetime, timedelta
import pandas as pd
from copy import deepcopy
from flask import render_template, request, jsonify, session, send_from_directory, make_response, abort
#from flask_jwt_extended import (create_access_token)

from config import Config
from packages import app, db_mysql, db, bcrypt, jwt, compress
from packages.Data.data import Data
from packages.Data.data_upload import DataUpload
from packages.Data.data_prep import DataPrep

@app.route( '/upload/deployments', methods=['POST', 'OPTIONS'] )
def upload_deployments():
	userpath = Config.UPLOAD_FOLDER
	file = request.files['file']

	# Preprocessing data
	try:
		data_upload = DataUpload( userpath)
		data_upload.readDataFrame( file)
	except Exception as error:
		response = jsonify( {"status": "Error"} )
		return response

	# Saving data
	data_upload.save( data_upload.data_df, userpath, 'deployments.csv')

	response = jsonify({"status": "Files uploaded"})
	return response

@app.route( '/upload/pickups', methods=['POST', 'OPTIONS'] )
def upload_pickups():
	userpath = Config.UPLOAD_FOLDER
	file = request.files['file']

	# Preprocessing data
	try:
		data_upload = DataUpload( userpath)
		data_upload.readDataFrame( file)
	except Exception as error:
		response = jsonify( {"status": "Error"} )
		return response

	# Saving data
	data_upload.save( data_upload.data_df, userpath, 'pickups.csv')

	response = jsonify({"status": "Files uploaded"})
	return response

@app.route( '/upload/rides', methods=['POST', 'OPTIONS'] )
def upload_rides():
	userpath = Config.UPLOAD_FOLDER
	file = request.files['file']

	# Preprocessing data
	try:
		data_upload = DataUpload( userpath)
		data_upload.readDataFrame( file)
	except Exception as error:
		response = jsonify( {"status": "Error"} )
		return response

	# Saving data
	data_upload.save( data_upload.data_df, userpath, 'rides.csv')

	response = jsonify({"status": "Files uploaded"})
	return response

@app.route( '/vehicles/<id>', methods=['GET'] )
def vehicle_performance(id):
	print('vehicle_id', id)

	if len(id) == 6:
		print('this is QR_code')
	else:
		print('this is vehicle_id')

	response = jsonify({"status": "OK"})
	return response

@app.route( "/" )
@app.route( '/<path:path>' )
def my_index(path=None):

	response = make_response( render_template( "index.html", token="Hello Flask+React" ) )
	# response.set_cookie('test_cookie', 'some_cookie', expires=datetime.utcnow() + timedelta(days=365))
	# print('request.cookie', request.cookies.get('test_cookie'))
	return response
