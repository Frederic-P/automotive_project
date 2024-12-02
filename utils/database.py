"""
    UTILITY CLASS TO INTERACT WITH THE DATABASE; DEALS WITH CONNECTION AND CRUD OPERATIONS
"""
import pymysql
import pymysql.cursors
import pandas as pd

class Database:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        """Establish a connection to the database."""
        if self.connection is None:
            try:
                self.connection = pymysql.connect(
                    host=self.host,
                    port = self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    cursorclass=pymysql.cursors.DictCursor
                )
                print("Connection established")
            except pymysql.MySQLError as e:
                print(f"Error connecting to MySQL: {e}")
                self.connection = None

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Connection closed")

     
    def start_transaction(self):
        """Start a new transaction."""
        self.connect()
        if self.connection:
            try:
                self.connection.begin()
            except pymysql.MySQLError as e:
                print(f"Error starting transaction: {e}")

    def commit_transaction(self):
        """Commit the current transaction."""
        if self.connection:
            try:
                self.connection.commit()
            except pymysql.MySQLError as e:
                print(f"Error committing transaction: {e}")
                self.connection.rollback()

    def rollback_transaction(self):
        """Rollback the current transaction."""
        if self.connection:
            try:
                self.connection.rollback()
                print("Transaction rolled back")
            except pymysql.MySQLError as e:
                print(f"Error rolling back transaction: {e}")

    def execute_query(self, query, params=None, return_rowid = False):
        """Execute a query and return the results."""
        self.connect()
        if self.connection:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                if return_rowid:
                    return cursor.lastrowid
                else:
                    return results
            except pymysql.MySQLError as e:
                print(f"Error executing query: {e}")
                return None

    def execute_update(self, query, params=None):
        """Execute an update/insert/delete query."""
        self.connect()
        if self.connection:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(query, params)
            except pymysql.MySQLError as e:
                print(f"Error executing update: {e}")
                self.connection.rollback()

    # def get_random_n_images(self, sitemap_id, n=100):
    #     """Get a random set of n image ids from the database.
    #     Since the volume is too big to randomly select on millions of
    #     records there first happens a random selection of the sitemap
    #     pk_id to randomly select within a randomly chosen subset."""
    #     random_images_in_subset = "SELECT pk_id, image_id, image_url FROM image_entries WHERE sitemap_source = %s AND state = %s AND is_duplicate = 0 ORDER BY rand() LIMIT %s"
    #     if self.connection:
    #         with self.connection.cursor() as cursor: 
    #             img_query_data = [sitemap_id, 'unprocessed', int(n)]
    #             cursor.execute(random_images_in_subset, img_query_data)
    #             results = cursor.fetchall()
    #             return results


    # def get_single_record(self, query, params=None):
    #     """Get a single record from the database."""
    #     results = self.execute_query(query, params)
    #     if results:
    #         return results[0]
    #     return None

    # def get_multiple_records(self, query, params=None):
    #     """Get multiple records from the database."""
    #     return self.execute_query(query, params)

    # def insert_record(self, query, params):
    #     """Insert a record into the database."""
    #     self.execute_update(query, params)

    # def update_record(self, query, params):
    #     """Update a record in the database."""
    #     self.execute_update(query, params)

    # def delete_record(self, query, params):
    #     """Delete a record from the database."""
    #     self.execute_update(query, params)

