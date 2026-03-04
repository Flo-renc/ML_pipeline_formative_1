Household Electric Power Consumption (Time Serires Analysis)
    A full-stack time series project built on the UCI Household Electric Power Consumption dataset. The prject cover exploratory data analysis, machine learning forcasting, database design (MongoDB and MySQL), a REST API, and an end-to-end prediction pipeline.

Table of Contents

Dataset Overview
Project Structure
Prerequisites
Installation
Running the Project

Task 1 — EDA & Model Training
Task 2 — Database Setup
Task 3 — REST API
Task 4 — Forecast Pipeline


API Reference
MongoDB Collections
MySQL Schema
Analytical Questions
Model Experiments
Environment Variables

Dataset Overview

Property                      Detail

Source                        UCI Machine Learning Repository
Time range                    December 2006 - November 2010
Granularity                   1-minute intervals
Total record                  ~2.07 million rows
Target Variable               Global_active_power (kilowatts)
Missing values                ~1.25% - habdled via forward/baward fill


Prerequisites

Python 3.10+
MongoDB 6+ running locally or via Docker
MySQL 8+ (for Task 2 SQL schema only — no Python driver required to run the rest)
Git

Installation
1. Clone repository
    git clone <your-repo-url>
    cd ML_pipeline_formative_1

2. Create a virtual enviroment
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate

3. Install dependencies
    pip install -r requirements.txt

4. Download Dataset
    Go to https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set?resource=download, download the zip, unzip it, and place household_power_consumption.txt inside the data/ folder.
5. Start MongoDB
    # Using Docker (easiest):
    docker run -d -p 27017:27017 --name mongo mongo:7

    # Or start your local MongoDB service:
    brew services start mongodb-community   # macOS
    sudo systemctl start mongod             # Linux

Analytical Questions

All questions are answered with visualisations and statistical interpretation in task1_eda/eda_analysis.py.

#Question                                                             Technique
Q1Does Global Active Power have an increasing or decreasing trend?    Linear regression on daily means
Q2What are the hourly and monthly seasonal patterns in power demand?  Group-by aggregation, bar charts
Q3Is consumption significantly different on weekends vs weekdays?     Welch's t-test, boxplot
Q4Are there significant lag/autocorrelation effects in the series?    Autocorrelation plot, lag correlation table (lagged features)Q5How do moving averages reveal underlying consumption patterns?                                      Rolling means at 1h / 1day / 1week windows (moving averages)