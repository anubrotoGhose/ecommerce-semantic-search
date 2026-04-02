import mysql.connector

conn = mysql.connector.connect(
    host="10.169.101.75",
    user="anubroto.g@10.169.101.65",
    password="Password@123",
    database="semantic_fashion_db"
)

cur = cnx.cursor()