"""
load_data.py
------------
Reads household_power_consumption.txt and loads
all data into 3 MySQL tables:
  - time_dimension
  - power_reading
  - sub_metering
"""

import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os
import time

# ── Load credentials from .env
load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', 3306)),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
)
cursor = conn.cursor()
print("Connected to MySQL ")

# STEP 1 — READ THE FILE

print("\nReading file...")

df = pd.read_csv(
    os.getenv('DATASET_PATH'),
    sep=';',
    na_values=['?', ''],
    low_memory=False
)
print(f"Rows loaded: {len(df):,}")

# STEP 2 — COMBINE DATE AND TIME INTO ONE COLUMN

df['recorded_at'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format='%d/%m/%Y %H:%M:%S'
)


# STEP 3 — DERIVE THE COLUMNS THAT DON'T EXIST IN THE FILE

df['hour'] = df['recorded_at'].dt.hour

# day_of_week → number from 1 (Monday) to 7 (Sunday)
# Python's isoweekday() returns exactly 1-7
df['day_of_week'] = df['recorded_at'].dt.weekday + 1

# month → number from 1 (January) to 12 (December)
df['month'] = df['recorded_at'].dt.month

# year → the full year
df['year'] = df['recorded_at'].dt.year

# is_weekend → 1 if Saturday or Sunday, 0 if weekday
# isoweekday() returns 6 for Saturday and 7 for Sunday
# so anything >= 6 is a weekend

df['is_weekend']  = (df['recorded_at'].dt.weekday >= 5).astype(int)

# STEP 4 — HANDLE MISSING VALUES

before = len(df)
df = df.dropna(subset=['Global_active_power'])
print(f"Rows dropped (missing power): {before - len(df):,}")

# For sub-metering: fill missing with 0
# Reasoning: if no reading was recorded, appliance was likely off
df['Sub_metering_1'] = df['Sub_metering_1'].fillna(0)
df['Sub_metering_2'] = df['Sub_metering_2'].fillna(0)
df['Sub_metering_3'] = df['Sub_metering_3'].fillna(0)

# For voltage and intensity: fill with column median
# Reasoning: median is a safe central estimate
for col in ['Global_reactive_power', 'Voltage', 'Global_intensity']:
    df[col] = df[col].fillna(df[col].median())

print(f"Rows ready to insert: {len(df):,}")


# STEP 5 — CLEAR TABLES FOR FRESH LOAD

print("\nClearing tables...")
cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
cursor.execute("TRUNCATE TABLE sub_metering")
cursor.execute("TRUNCATE TABLE power_reading")
cursor.execute("TRUNCATE TABLE time_dimension")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
conn.commit()
print("Tables cleared")

# STEP 6 — INSERT ROW BY ROW INTO ALL 3 TABLES

print(f"\nInserting {len(df):,} rows...")

BATCH_SIZE = 500   # commit to disk every 500 rows
count      = 0
start      = time.time()

for _, row in df.iterrows():

    # Insert into time_dimension 
    # These are all the WHEN columns
    cursor.execute("""
        INSERT INTO time_dimension
            (recorded_at, hour, day_of_week, month, year, is_weekend)
        VALUES
            (%s, %s, %s, %s, %s, %s)
    """, (
        row['recorded_at'],      
        int(row['hour']),        
        int(row['day_of_week']), 
        int(row['month']),       
        int(row['year']),        
        int(row['is_weekend'])   
    ))

    # MySQL gives back the auto-generated time_id for this row
    time_id = cursor.lastrowid

    #  Insert into power_reading 
    # These are the HOW MUCH columns
    # We pass time_id so this reading is linked to its timestamp
    cursor.execute("""
        INSERT INTO power_reading
            (time_id, global_active_power, global_reactive_power,
             voltage, global_intensity)
        VALUES
            (%s, %s, %s, %s, %s)
    """, (
        time_id,                             
        float(row['Global_active_power']),    
        float(row['Global_reactive_power']),  
        float(row['Voltage']),                
        float(row['Global_intensity'])        
    ))

   
    reading_id = cursor.lastrowid

    #  Insert into sub_metering 
    # These are the WHERE columns (which appliance used what)
    # We pass reading_id so this is linked to its power reading
    cursor.execute("""
        INSERT INTO sub_metering
            (reading_id, kitchen, laundry, water_heater_ac)
        VALUES
            (%s, %s, %s, %s)
    """, (
        reading_id,                      
        float(row['Sub_metering_1']),    
        float(row['Sub_metering_2']),    
        float(row['Sub_metering_3'])     
    ))

    count += 1

    # Commit every 500 rows and show progress
    if count % BATCH_SIZE == 0:
        conn.commit()
        elapsed   = time.time() - start
        rate      = count / elapsed
        remaining = (len(df) - count) / rate
        pct       = (count / len(df)) * 100
        print(f"  [{pct:5.1f}%] {count:,} rows | "
              f"{rate:,.0f} rows/sec | "
              f"~{remaining/60:.1f} min left")

# Final commit for leftover rows
conn.commit()
print(f"\nInserted {count:,} rows in {(time.time()-start)/60:.1f} minutes ✅")

# STEP 7 — VERIFY
print("\nVerification:")
for table in ['time_dimension', 'power_reading', 'sub_metering']:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"  {table:<20} → {cursor.fetchone()[0]:>10,} rows")

cursor.execute("""
    SELECT MIN(recorded_at), MAX(recorded_at)
    FROM time_dimension
""")
first, last = cursor.fetchone()
print(f"\n  First record : {first}")
print(f"  Last record  : {last}")

cursor.close()
conn.close()
print("\nDone")