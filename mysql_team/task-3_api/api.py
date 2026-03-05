"""
app.py
FastAPI for MySQL power consumption database.

Endpoints:
  POST   /sql/readings              → create a new reading
  GET    /sql/readings              → get all readings (paginated)
  GET    /sql/readings/latest       → get the most recent reading
  GET    /sql/readings/range        → get readings by date range
  PUT    /sql/readings/{id}         → update a reading
  DELETE /sql/readings/{id}         → delete a reading
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime
import os

#  Load .env 
load_dotenv()

app = FastAPI(
    title="Power Consumption API",
    description="CRUD API for household electric power consumption data",
    version="1.0.0"
)


def get_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

# PYDANTIC MODELS — define shape of request bodies
# FastAPI validates incoming data against these automatically

class ReadingCreate(BaseModel):
    """All fields required when creating a new reading."""
    recorded_at:            str    
    global_active_power:    float
    global_reactive_power:  float
    voltage:                float
    global_intensity:       float
    kitchen:                float
    laundry:                float
    water_heater_ac:        float

class ReadingUpdate(BaseModel):
    """All fields optional — only send what you want to change."""
    global_active_power:    Optional[float] = None
    global_reactive_power:  Optional[float] = None
    voltage:                Optional[float] = None
    global_intensity:       Optional[float] = None
    kitchen:                Optional[float] = None
    laundry:                Optional[float] = None
    water_heater_ac:        Optional[float] = None


def format_row(row):
    """Converts a raw DB row tuple into a clean dictionary."""
    return {
        "reading_id":             row[0],
        "recorded_at":            str(row[1]),
        "hour":                   row[2],
        "day_of_week":            row[3],
        "month":                  row[4],
        "year":                   row[5],
        "is_weekend":             bool(row[6]),
        "global_active_power":    float(row[7])  if row[7]  else None,
        "global_reactive_power":  float(row[8])  if row[8]  else None,
        "voltage":                float(row[9])  if row[9]  else None,
        "global_intensity":       float(row[10]) if row[10] else None,
        "kitchen":                float(row[11]) if row[11] else None,
        "laundry":                float(row[12]) if row[12] else None,
        "water_heater_ac":        float(row[13]) if row[13] else None,
    }

# Reusable SELECT joining all 3 tables
BASE_SELECT = """
    SELECT
        pr.reading_id,
        td.recorded_at,
        td.hour,
        td.day_of_week,
        td.month,
        td.year,
        td.is_weekend,
        pr.global_active_power,
        pr.global_reactive_power,
        pr.voltage,
        pr.global_intensity,
        sm.kitchen,
        sm.laundry,
        sm.water_heater_ac
    FROM power_reading pr
    JOIN time_dimension td ON pr.time_id    = td.time_id
    JOIN sub_metering   sm ON sm.reading_id = pr.reading_id
"""


@app.get("/sql/readings/latest", summary="Get the most recent reading")
def get_latest():
    """
    Returns the single most recent power reading.
    Example: GET http://localhost:5000/sql/readings/latest
    """
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(BASE_SELECT + "ORDER BY td.recorded_at DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No records found")
        return format_row(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# GET /sql/readings/range?start=2007-01-01&end=2007-01-31

@app.get("/sql/readings/range", summary="Get readings by date range")
def get_by_range(
    start: str = Query(..., description="Start date e.g. 2007-01-01"),
    end:   str = Query(..., description="End date   e.g. 2007-01-31")
):
    """
    Returns all readings between two dates (max 1000 rows).
    Example: GET http://localhost:5000/sql/readings/range?start=2007-01-01&end=2007-01-31
    """
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(BASE_SELECT + """
            WHERE td.recorded_at BETWEEN %s AND %s
            ORDER BY td.recorded_at ASC
            LIMIT 1000
        """, (start, end))
        rows = cursor.fetchall()
        return {
            "count":   len(rows),
            "start":   start,
            "end":     end,
            "results": [format_row(r) for r in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# GET /sql/readings?page=1&limit=50

@app.get("/sql/readings", summary="Get all readings (paginated)")
def get_all(
    page:  int = Query(1,  description="Page number"),
    limit: int = Query(50, description="Rows per page")
):
    """
    Returns paginated readings.
    Example: GET http://localhost:5000/sql/readings?page=1&limit=50
    """
    try:
        offset = (page - 1) * limit
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(BASE_SELECT + """
            ORDER BY td.recorded_at ASC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cursor.fetchall()
        return {
            "page":    page,
            "limit":   limit,
            "count":   len(rows),
            "results": [format_row(r) for r in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/sql/readings", status_code=201, summary="Create a new reading")
def create_reading(body: ReadingCreate):
    """
    Creates a new reading across all 3 tables.
    Order: time_dimension → power_reading → sub_metering
    """
    try:
        conn   = get_db()
        cursor = conn.cursor()
        dt     = datetime.fromisoformat(body.recorded_at)

        # Step 1 — time_dimension (WHEN)
        cursor.execute("""
            INSERT INTO time_dimension
                (recorded_at, hour, day_of_week, month, year, is_weekend)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            dt,
            dt.hour,
            dt.isoweekday(),
            dt.month,
            dt.year,
            1 if dt.isoweekday() >= 6 else 0
        ))
        time_id = cursor.lastrowid

        # Step 2 — power_reading (HOW MUCH)
        cursor.execute("""
            INSERT INTO power_reading
                (time_id, global_active_power, global_reactive_power,
                 voltage, global_intensity)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            time_id,
            body.global_active_power,
            body.global_reactive_power,
            body.voltage,
            body.global_intensity
        ))
        reading_id = cursor.lastrowid

        # Step 3 — sub_metering (WHERE)
        cursor.execute("""
            INSERT INTO sub_metering
                (reading_id, kitchen, laundry, water_heater_ac)
            VALUES (%s, %s, %s, %s)
        """, (reading_id, body.kitchen, body.laundry, body.water_heater_ac))

        conn.commit()
        return {
            "message":    "Reading created successfully",
            "reading_id": reading_id,
            "time_id":    time_id
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# PUT /sql/readings/{reading_id}

@app.put("/sql/readings/{reading_id}", summary="Update a reading")
def update_reading(reading_id: int, body: ReadingUpdate):
    """
    Updates power values for an existing reading.
    Only include the fields you want to change.
    Example: PUT http://localhost:5000/sql/readings/1
    """
    try:
        conn   = get_db()
        cursor = conn.cursor()

        # Check reading exists
        cursor.execute(
            "SELECT time_id FROM power_reading WHERE reading_id = %s",
            (reading_id,)
        )
        if not cursor.fetchone():
            raise HTTPException(
                status_code=404,
                detail=f"reading_id {reading_id} not found"
            )

        # Update power_reading
        power_fields  = ['global_active_power', 'global_reactive_power',
                         'voltage', 'global_intensity']
        power_updates = {f: getattr(body, f) for f in power_fields
                         if getattr(body, f) is not None}
        if power_updates:
            set_clause = ", ".join([f"{k} = %s" for k in power_updates])
            cursor.execute(
                f"UPDATE power_reading SET {set_clause} WHERE reading_id = %s",
                list(power_updates.values()) + [reading_id]
            )

        # Update sub_metering
        sub_fields  = ['kitchen', 'laundry', 'water_heater_ac']
        sub_updates = {f: getattr(body, f) for f in sub_fields
                       if getattr(body, f) is not None}
        if sub_updates:
            set_clause = ", ".join([f"{k} = %s" for k in sub_updates])
            cursor.execute(
                f"UPDATE sub_metering SET {set_clause} WHERE reading_id = %s",
                list(sub_updates.values()) + [reading_id]
            )

        conn.commit()
        return {"message": "Reading updated successfully", "reading_id": reading_id}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# DELETE /sql/readings/{reading_id}

@app.delete("/sql/readings/{reading_id}", summary="Delete a reading")
def delete_reading(reading_id: int):
    """
    Deletes a reading from all 3 tables.
    Delete order: sub_metering → power_reading → time_dimension
    Example: DELETE http://localhost:5000/sql/readings/1
    """
    try:
        conn   = get_db()
        cursor = conn.cursor()

        # Get time_id before deleting
        cursor.execute(
            "SELECT time_id FROM power_reading WHERE reading_id = %s",
            (reading_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"reading_id {reading_id} not found"
            )
        time_id = row[0]

        # Delete in reverse FK order
        cursor.execute("DELETE FROM sub_metering  WHERE reading_id = %s", (reading_id,))
        cursor.execute("DELETE FROM power_reading  WHERE reading_id = %s", (reading_id,))
        cursor.execute("DELETE FROM time_dimension WHERE time_id    = %s", (time_id,))

        conn.commit()
        return {"message": "Reading deleted successfully", "reading_id": reading_id}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()