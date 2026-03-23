# AI in Autonomous Vehicles

DriveLearn is a DBMS-focused autonomous vehicle benchmarking project built with Flask and SQLite. It compares AI models across realistic driving scenarios such as rain, fog, and sunny highway conditions, then surfaces the results through a dashboard and SQL-based analytics.

## Project Contents

- `app.py` - Flask application, schema creation, seed logic, live simulation routes, and analytics APIs
- `query_db.py` - helper script for running SQL against the SQLite database
- `drivelearn.db` - project database used by the application
- `templates/index.html` - dashboard UI
- project reports, presentations, and chapter notes used during development

## Features

- normalized relational schema for `AI_Models`, `Scenarios`, `Vehicles`, `Test_Runs`, `Frames`, and `Detections`
- SQLite constraints including `UNIQUE`, `CHECK`, and foreign keys
- seeded benchmark data plus additional live simulation records
- analytics for confidence, speed, detections, scenarios, and latest runs
- interactive dashboard for vehicle selection, scenario testing, and result visualization

## Tech Stack

- Python
- Flask
- SQLite
- HTML, Tailwind CSS, JavaScript

## Run Locally

1. Install Python 3.
2. Install dependencies:

```powershell
pip install Flask
```

3. Start the app:

```powershell
python app.py
```

4. Open:

```text
http://127.0.0.1:5000
```

## Database Overview

The system stores:

- AI models and their architecture and accuracy
- driving scenarios with weather, light, road type, and difficulty
- vehicles mapped to AI models
- test runs for each vehicle-scenario pair
- frames captured during each run
- detections generated from each frame

## Repository Notes

- The database file is included so the current project progress is preserved.
- Generated cache files and local server logs are ignored.

## Author

Project repository prepared from the current DriveLearn workspace for GitHub publication.
