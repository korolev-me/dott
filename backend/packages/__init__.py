import os
from flask import Flask
from flask_compress import Compress
from flask_cors import CORS
from flask_mysqldb import MySQL
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

from config import Config

app = Flask( __name__, template_folder='./static', static_folder='./static',
             root_path=Config.ROOT_FOLDER )  # , static_url_path=''
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['DATA_AUX_FOLDER'] = Config.DATA_AUX_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['MYSQL_USER'] = Config.MYSQL_USER
app.config['MYSQL_PASSWORD'] = Config.MYSQL_PASSWORD
app.config['MYSQL_DB'] = 'demand_forecast'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['JWT_SECRET_KEY'] = 'secret'
app.config['COMPRESS_ALGORITHM'] = 'gzip'

app.secret_key = os.urandom( 24 )

db_mysql = MySQL( app )
db = SQLAlchemy( app )
bcrypt = Bcrypt( app )
jwt = JWTManager( app )
compress = Compress( app )
_ = CORS( app, expose_headers='Authorization' )

from packages import routes

print('Config.ROOT_FOLDER', Config.ROOT_FOLDER)