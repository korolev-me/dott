import os

os.environ["MYSQL_USER"] = "root"
os.environ["MYSQL_PASSWORD"] = "dott_password"

class Config( object ):
	ROOT_FOLDER = os.getcwd()
	UPLOAD_FOLDER = os.path.join( ROOT_FOLDER, 'data_upload' )
	DATA_AUX_FOLDER = os.path.join( ROOT_FOLDER, 'data_aux' )
	MYSQL_USER = os.environ.get( 'MYSQL_USER' )
	MYSQL_PASSWORD = os.environ.get( 'MYSQL_PASSWORD' )
	SQLALCHEMY_DATABASE_URI = 'mysql://' + str(MYSQL_USER) + ':' + str(MYSQL_PASSWORD) + '@localhost/dott'
	SQLALCHEMY_TRACK_MODIFICATIONS = False