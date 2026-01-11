import sqlite3
con = sqlite3.connect("finance_runs.db")
rows = con.execute("SELECT id, created_at, finance_health_score FROM runs ORDER BY id DESC LIMIT 5").fetchall()
print(rows)
con.close()
