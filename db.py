import psycopg
from pgvector.psycopg import register_vector

from config import DATABASE_URL, ROOT

SCHEMA = (ROOT / "schema.sql").read_text()


def connect() -> psycopg.Connection:
    """Open a connection, make sure the schema exists and teach psycopg the vector type."""
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute(SCHEMA)
    # Must run after CREATE EXTENSION, otherwise the vector type does not exist yet.
    register_vector(conn)
    return conn
