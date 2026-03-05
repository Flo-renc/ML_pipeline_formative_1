-- Get the most recent power reading with full details
SELECT td.recorded_at, td.hour, td.day_of_week, td.is_weekend, pr.global_active_power, pr.voltage, sm.kitchen, sm.laundry, sm.water_heater_ac
FROM
    power_reading pr
    JOIN time_dimension td ON pr.time_id = td.time_id
    JOIN sub_metering sm ON sm.reading_id = pr.reading_id
ORDER BY td.recorded_at DESC
LIMIT 1;

-- Get all readings for January 2007
-- Shows total house consumption + appliance breakdown per minute
SELECT td.recorded_at, td.hour, td.day_of_week, pr.global_active_power, pr.voltage, sm.kitchen, sm.laundry, sm.water_heater_ac
FROM
    power_reading pr
    JOIN time_dimension td ON pr.time_id = td.time_id
    JOIN sub_metering sm ON sm.reading_id = pr.reading_id
WHERE
    td.recorded_at BETWEEN '2007-01-01 00:00:00' AND '2007-01-31 23:59:59'
ORDER BY td.recorded_at ASC;

SELECT
    td.day_of_week,
    td.hour,
    ROUND(
        AVG(pr.global_active_power),
        3
    ) AS avg_active_power,
    ROUND(
        MAX(pr.global_active_power),
        3
    ) AS peak_active_power,
    ROUND(AVG(sm.water_heater_ac), 3) AS avg_water_heater,
    ROUND(AVG(sm.kitchen), 3) AS avg_kitchen,
    COUNT(*) AS total_readings
FROM
    power_reading pr
    JOIN time_dimension td ON pr.time_id = td.time_id
    JOIN sub_metering sm ON sm.reading_id = pr.reading_id
GROUP BY
    td.day_of_week,
    td.hour
ORDER BY td.day_of_week ASC, td.hour ASC;

--- Analyticaln Question query
SELECT
    DATE(td.recorded_at) AS reading_date,
    ROUND(
        AVG(pr.global_active_power),
        3
    ) AS daily_avg_power,
    ROUND(
        AVG(AVG(pr.global_active_power)) OVER (
            ORDER BY DATE(td.recorded_at) ROWS BETWEEN 6 PRECEDING
                AND CURRENT ROW
        ),
        3
    ) AS moving_avg_7day
FROM
    power_reading pr
    JOIN time_dimension td ON pr.time_id = td.time_id
GROUP BY
    DATE(td.recorded_at)
ORDER BY reading_date ASC;