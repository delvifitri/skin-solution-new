import mysql.connector
import os

db = mysql.connector.connect(
    host=os.getenv("DB_HOST", default="localhost"),
    user=os.getenv("DB_USER", default="root"),
    password=os.getenv("DB_PASS", default=""),
    database=os.getenv("DB_NAME", default="fd_reviews")
)