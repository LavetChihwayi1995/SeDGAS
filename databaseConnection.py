app = Flask(__name__)

# ------------------------------------------------------------------------------------------------
# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'SeGas'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql=MySQL(app)