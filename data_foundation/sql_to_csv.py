import os
import sqlite3
import duckdb
import mysql.connector
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime

# ---------------------------------------------------
# ENV
# ---------------------------------------------------
load_dotenv()

MYSQL_HOST = os.getenv("mysql_um_db_host", "localhost")
MYSQL_PORT = int(os.getenv("mysql_um_db_port", 3306))
MYSQL_USER = os.getenv("mysql_um_db_user", "root")
MYSQL_PASSWORD = os.getenv("mysql_um_db_password", "")
MYSQL_DATABASE = os.getenv("mysql_um_db_name", "USER_MANAGEMENT")

OUTPUT_DIR = "binary_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="MySQL → SQLite / DuckDB Export Service",
    version="3.0.0"
)

# ---------------------------------------------------
# Shared Schema Mapper
# ---------------------------------------------------
def map_mysql_to_sqlite_type(mysql_type: str) -> str:
    t = mysql_type.lower()
    if "int" in t:
        return "INTEGER"
    if "float" in t or "double" in t or "decimal" in t:
        return "REAL"
    if "bool" in t:
        return "INTEGER"
    if "char" in t or "text" in t:
        return "TEXT"
    return "TEXT"

# ---------------------------------------------------
# Core Export Logic (Generic)
# ---------------------------------------------------
def export_mysql_to_binary(target: str):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ext = "db" if target == "sqlite" else "duckdb"
    out_path = os.path.join(
        OUTPUT_DIR, f"{MYSQL_DATABASE}_{timestamp}.{ext}"
    )

    try:
        mysql_conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

    mysql_cursor = mysql_conn.cursor()
    mysql_cursor.execute("SHOW TABLES")
    tables = [row[0] for row in mysql_cursor.fetchall()]

    # ---------------------------------------------------
    # Target DB Init
    # ---------------------------------------------------
    if target == "sqlite":
        target_conn = sqlite3.connect(out_path)
        execute = target_conn.execute
    else:
        target_conn = duckdb.connect(out_path)
        execute = target_conn.execute

    # ---------------------------------------------------
    # Table Copy
    # ---------------------------------------------------
    for table in tables:
        mysql_cursor.execute(f"DESCRIBE `{table}`")
        columns = mysql_cursor.fetchall()

        col_defs = []
        col_names = []

        for col in columns:
            name = col[0]
            mysql_type = col[1]
            col_defs.append(f'"{name}" {map_mysql_to_sqlite_type(mysql_type)}')
            col_names.append(name)

        create_stmt = f"""
        CREATE TABLE "{table}" (
            {", ".join(col_defs)}
        )
        """
        execute(create_stmt)

        mysql_data_cursor = mysql_conn.cursor()
        mysql_data_cursor.execute(f"SELECT * FROM `{table}`")

        placeholders = ",".join(["?"] * len(col_names))
        insert_stmt = f"""
        INSERT INTO "{table}" ({",".join(col_names)})
        VALUES ({placeholders})
        """

        for row in mysql_data_cursor:
            execute(insert_stmt, row)

        mysql_data_cursor.close()
        target_conn.commit()

    mysql_cursor.close()
    mysql_conn.close()
    target_conn.close()

    return out_path

# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/export/mysql-to-sqlite")
def export_sqlite():
    path = export_mysql_to_binary("sqlite")
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=os.path.basename(path)
    )

@app.post("/export/mysql-to-duckdb")
def export_duckdb():
    path = export_mysql_to_binary("duckdb")
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=os.path.basename(path)
    )
