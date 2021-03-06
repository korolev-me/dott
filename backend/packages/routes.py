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

import time

@app.route( '/upload/deployments', methods=['POST', 'OPTIONS'] )
def upload_deployments():
	file = request.files['file']

	print('upload_deployments')
	# Preprocessing data
	try:
		data_upload = DataUpload()
		data_upload.readDataFrame(file)
	except Exception as error:
		response = jsonify( {"status": "Error"} )
		return response

	# Saving data
	start_time = time.time()
	data_upload.data_df = data_upload.data_df.where(pd.notnull(data_upload.data_df), None)
	data_l = data_upload.data_df.values.tolist()
	cur = db_mysql.connection.cursor()
	sql = "INSERT IGNORE INTO deployments VALUES (%s, %s, %s, %s)"
	cur.executemany(sql, data_l)
	db_mysql.connection.commit()
	print("--- %s seconds ---" % (time.time() - start_time))

	response = jsonify({"status": "Files uploaded"})
	return response

@app.route( '/upload/pickups', methods=['POST', 'OPTIONS'] )
def upload_pickups():
	file = request.files['file']

	print('upload_pickups')
	# Preprocessing data
	try:
		data_upload = DataUpload()
		data_upload.readDataFrame(file)
	except Exception as error:
		response = jsonify( {"status": "Error"} )
		return response

	# Saving data
	start_time = time.time()
	data_upload.data_df = data_upload.data_df.where(pd.notnull(data_upload.data_df), None)
	data_l = data_upload.data_df.values.tolist()
	cur = db_mysql.connection.cursor()
	sql = "INSERT IGNORE INTO pickups VALUES (%s, %s, %s, %s, %s)"
	cur.executemany(sql, data_l)
	db_mysql.connection.commit()
	print("--- %s seconds ---" % (time.time() - start_time))

	response = jsonify({"status": "Files uploaded"})
	return response

@app.route( '/upload/rides', methods=['POST', 'OPTIONS'] )
def upload_rides():
	file = request.files['file']

	# Preprocessing data
	try:
		data_upload = DataUpload()
		data_upload.readDataFrame(file)
	except Exception as error:
		response = jsonify( {"status": "Error"} )
		return response

	# Saving data
	start_time = time.time()
	data_upload.data_df = data_upload.data_df.where(pd.notnull(data_upload.data_df), None)
	data_l = data_upload.data_df.values.tolist()
	cur = db_mysql.connection.cursor()
	sql = "INSERT IGNORE INTO rides VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
	cur.executemany(sql, data_l)
	db_mysql.connection.commit()
	print("--- %s seconds ---" % (time.time() - start_time))

	response = jsonify({"status": "Files uploaded"})
	return response

@app.route( '/vehicles/<id>', methods=['GET'] )
def vehicle_performance(id):
	print('vehicle_id', id)

	if len(id) == 6:
		print('this is QR_code')
	else:
		print('this is vehicle_id')

	vehicle_id = id

	request_deployment = """
	SELECT 
		'deployment' as type,
		COUNT(ride_id) as ride_count,
		IFNULL(AVG(gross_amount), 0) as gross_amount,
		FLOOR(IFNULL(AVG(time_to_ride), 0)) as time_to_ride,
		IFNULL(AVG(travel_distance), 0) as travel_distance
	FROM cycles_params 
	WHERE vehicle_id = %s 
	GROUP BY time_deployment
	ORDER BY time_deployment desc
	;
	"""

	request_ride = """
	SELECT 
		'ride' as type,
		1 as ride_count,
		gross_amount,
		time_to_ride,
		travel_distance
	FROM cycles_params cp
	JOIN (
		SELECT 
			vehicle_id,
			time_deployment
		FROM cycles
		WHERE vehicle_id = %s 
		ORDER BY time_deployment desc
		LIMIT 1) as lc
	ON ((cp.vehicle_id = lc.vehicle_id) AND (cp.time_deployment = lc.time_deployment))
	ORDER BY gross_amount desc
	LIMIT 5
	;
	"""

	cur = db_mysql.connection.cursor()
	cur.execute(request_deployment, [vehicle_id])
	db_mysql.connection.commit()
	resp_deployment = cur.fetchall()

	cur.execute(request_ride, [vehicle_id])
	db_mysql.connection.commit()
	resp_ride = cur.fetchall()

	resp_sql = (resp_ride+resp_deployment)[:5]
	response = jsonify({"resp_sql": resp_sql})
	return response

@app.route( "/" )
@app.route( '/<path:path>' )
def my_index(path=None):

	response = make_response( render_template( "index.html", token="Hello Flask+React" ) )
	# response.set_cookie('test_cookie', 'some_cookie', expires=datetime.utcnow() + timedelta(days=365))
	# print('request.cookie', request.cookies.get('test_cookie'))
	return response
