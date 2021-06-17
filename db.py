from getpass import getpass
from mysql.connector import connect, Error


class DataBase(object):
    __instance = None

    def __init__(self):
        try:
            self.connection = connect(
                host="localhost",
                user='',
                password='',
                database="notes",
                port='3306')
        except Error as e:
            print(e)

    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = DataBase()
        return cls.__instance

    def save_user(self, user):
        insert_user_query = "INSERT INTO users (user_email, access_token, refresh_token, checked_label_id) VALUES (" \
                            "%s, %s, %s, %s); "
        params = (user['user_email'],
                  user['access_token'],
                  user['refresh_token'],
                  user['checked_label_id'])
        with self.connection.cursor() as cursor:
            cursor.execute(insert_user_query,
                           params)
            self.connection.commit()

    def get_all(self):
        select_query = "SELECT * FROM users;"
        with self.connection.cursor() as cursor:
            cursor.execute(select_query)
            return cursor.fetchall()

    def delete_user(self, access_token):
        delete_q = "DELETE FROM users WHERE access_token = %s;"
        with self.connection.cursor() as cursor:
            cursor.execute(delete_q, (access_token,))
            self.connection.commit()

    def update_user_access_token(self, access_token, email):
        upd_q = "UPDATE users set access_token=%s where user_email=%s"
        with self.connection.cursor() as cursor:
            cursor.execute(upd_q, (access_token, email))
            self.connection.commit()

    def update_user(self, access_token, refresh_token, email):
        upd_q = "UPDATE users SET access_token=%s, refresh_token=%s WHERE user_email=%s"
        with self.connection.cursor() as cursor:
            cursor.execute(upd_q, (access_token, refresh_token, email))
            self.connection.commit()


db = DataBase.getInstance()
