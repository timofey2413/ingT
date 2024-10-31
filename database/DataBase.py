import sqlite3 as sl
 
con = sl.connect('example.db')
with con:
    con.execute('''
        CREATE TABLE users(
            id INTEGER PRIMARY KEY, 
            name TEXT
);
    ''')
con.close()