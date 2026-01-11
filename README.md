# Zoro AI – AWS + MySQL (RDS) Setup

This project keeps the same URLs you already use:
- `GET /dashboard` → dashboard page (reads latest saved run from the DB)
- `GET /api/latest` → latest saved run
- `POST /api/predict` → run prediction + store into DB

The only change is the **storage layer**: on AWS you should use **MySQL (RDS)** instead of a local `.db` file.

## 1) Create MySQL on AWS (RDS)
1. AWS Console → **RDS** → **Create database**
2. Engine: **MySQL**
3. Public access: ... (for testing) or private (recommended) with EC2 in same VPC
4. Note the values:
   - Endpoint (HOST)
   - Port (usually 3306)
   - Master username
   - Master password
   - DB name (create one, e.g. `zoro_finance`)

## 2) Connect in MySQL Workbench (to see your schema)
1. MySQL Workbench → **+** (New Connection)
2. Hostname: **RDS endpoint**
3. Port: **3306**
4. Username: your RDS master username
5. Password: store in vault
6. Test Connection
7. After connect: **Schemas panel** (left) → refresh → you will see `zoro_finance`.

The table `runs` will be created automatically by the app the first time it starts.

## 3) Set environment variables on AWS
The app reads DB settings from **either** `DATABASE_URL` **or** the `MYSQL_*` variables.

### Option A (recommended): DATABASE_URL
Example:
```
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/zoro_finance
```

### Option B: MYSQL_* vars
```
MYSQL_HOST=...
MYSQL_PORT=3306
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_DBNAME=zoro_finance
```

## 4) Deploy
### Option 1: EC2 (simple)
1. Launch an EC2 instance (Ubuntu)
2. Install Docker
3. Copy this project folder to the instance
4. Build + run:
```
docker build -t zoro-ai .
docker run -p 5000:5000 \
  -e DATABASE_URL="mysql+pymysql://USER:PASSWORD@HOST:3306/zoro_finance" \
  -e DATASET_PATH="/app/data.csv" \
  zoro-ai
```
5. Open:
- `http://EC2_PUBLIC_IP:5000/` (login)
- `http://EC2_PUBLIC_IP:5000/dashboard`

### Option 2: Elastic Beanstalk
- Use the included `Procfile` (Gunicorn) and set environment variables in the EB console.

## 5) Dataset for training (optional)
Set `DATASET_PATH` to your CSV on the server. If not set, the service uses a small built-in dataset so the API still runs.

## Notes
- The dashboard **does not call `/api/predict` automatically** anymore. It calls only `/api/latest`, so it always reflects what is saved in the DB.
- To create a new run, call `POST /api/predict` from your form/backend.
