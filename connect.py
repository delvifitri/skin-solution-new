import mysql.connector
import os

from mysql.connector.errors import OperationalError

db = mysql.connector.connect(
    host=os.getenv("DB_HOST", default="localhost"),
    user=os.getenv("DB_USER", default="root"),
    password=os.getenv("DB_PASS", default=""),
    database=os.getenv("DB_NAME", default="fd_reviews")
)

def cursor(dictionary=False):
    try:
        return db.cursor(dictionary=dictionary)
    except OperationalError as e:
        db.reconnect()
        return db.cursor(dictionary=dictionary)