from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "drivelearn.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS AI_Models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_name TEXT NOT NULL UNIQUE,
    architecture TEXT NOT NULL,
    accuracy_score REAL NOT NULL CHECK (accuracy_score BETWEEN 0.0 AND 1.0)
);

CREATE TABLE IF NOT EXISTS Scenarios (
    scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
    weather_type TEXT NOT NULL CHECK (weather_type IN ('Rain', 'Fog', 'Sunny')),
    light_level TEXT NOT NULL CHECK (light_level IN ('Day', 'Night')),
    road_type TEXT NOT NULL CHECK (road_type IN ('Highway', 'Urban')),
    difficulty TEXT NOT NULL CHECK (difficulty IN ('Low', 'Medium', 'High', 'Critical'))
);

CREATE TABLE IF NOT EXISTS Vehicles (
    vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    make_model TEXT NOT NULL,
    ai_model_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (ai_model_id) REFERENCES AI_Models(model_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS Test_Runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER NOT NULL,
    scenario_id INTEGER NOT NULL,
    test_date TEXT NOT NULL,
    avg_speed REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (vehicle_id) REFERENCES Vehicles(vehicle_id) ON DELETE RESTRICT,
    FOREIGN KEY (scenario_id) REFERENCES Scenarios(scenario_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS Frames (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    image_path TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES Test_Runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    object_type TEXT NOT NULL,
    confidence_score NUMERIC NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 0.999),
    FOREIGN KEY (frame_id) REFERENCES Frames(frame_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_test_runs_scenario_id ON Test_Runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_detections_confidence_score ON Detections(confidence_score);
"""

MODEL_SEED = [
    ("DriveNet v1.0", "Hybrid CNN", 0.78),
    ("DriveNet v1.4", "Vision Transformer", 0.86),
    ("Sentinel AV v2.1", "Multimodal Fusion", 0.91),
]

SCENARIO_SEED = [
    ("Rain", "Night", "Highway", "High"),
    ("Sunny", "Day", "Highway", "Low"),
    ("Fog", "Day", "Urban", "Critical"),
]

VEHICLE_SEED = [
    ("Orion EV Fleet One", "DriveNet v1.0", "Active"),
    ("Orion EV Fleet Two", "DriveNet v1.4", "Active"),
    ("Atlas Urban Shuttle", "Sentinel AV v2.1", "Standby"),
]

OBJECT_TYPES = ["Pedestrian", "Car", "Cyclist", "Truck", "Barrier", "Signal Cone"]
WEATHER_PENALTY = {"Sunny": 0.05, "Rain": -0.08, "Fog": -0.12}
DIFFICULTY_PENALTY = {"Low": 0.04, "Medium": 0.0, "High": -0.06, "Critical": -0.12}
ROAD_SPEED = {"Highway": (58, 78), "Urban": (24, 42)}


app = Flask(__name__)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scenario_label(row: sqlite3.Row | dict[str, Any]) -> str:
    weather = row["weather_type"]
    light = row["light_level"]
    road = row["road_type"]

    if weather == "Rain" and light == "Night":
        return "Rainy Night"
    if weather == "Sunny" and light == "Day":
        return "Sunny Day"
    if weather == "Fog" and road == "Urban":
        return "Foggy Intersection"
    return f"{weather} {light} {road}"


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        run_normalization_check(conn)
        seed_data(conn)


def run_normalization_check(conn: sqlite3.Connection) -> bool:
    environmental_columns = {"weather_type", "light_level", "road_type", "difficulty"}
    checked_tables = ["AI_Models", "Vehicles", "Test_Runs", "Frames", "Detections"]

    for table in checked_tables:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if columns & environmental_columns:
            raise ValueError(f"Normalization failure: environmental metadata leaked into {table}")
    return True


def seed_data(conn: sqlite3.Connection) -> None:
    existing_models = conn.execute("SELECT COUNT(*) AS count FROM AI_Models").fetchone()["count"]
    if existing_models:
        return

    conn.executemany(
        """
        INSERT INTO AI_Models (version_name, architecture, accuracy_score)
        VALUES (?, ?, ?)
        """,
        MODEL_SEED,
    )

    conn.executemany(
        """
        INSERT INTO Scenarios (weather_type, light_level, road_type, difficulty)
        VALUES (?, ?, ?, ?)
        """,
        SCENARIO_SEED,
    )

    model_lookup = {
        row["version_name"]: row["model_id"]
        for row in conn.execute("SELECT model_id, version_name FROM AI_Models").fetchall()
    }
    conn.executemany(
        """
        INSERT INTO Vehicles (make_model, ai_model_id, status)
        VALUES (?, ?, ?)
        """,
        [(name, model_lookup[version], status) for name, version, status in VEHICLE_SEED],
    )

    rng = random.Random(42)
    vehicles = conn.execute(
        """
        SELECT v.vehicle_id, v.make_model, m.accuracy_score
        FROM Vehicles v
        JOIN AI_Models m ON m.model_id = v.ai_model_id
        ORDER BY v.vehicle_id
        """
    ).fetchall()
    scenarios = conn.execute(
        """
        SELECT scenario_id, weather_type, light_level, road_type, difficulty
        FROM Scenarios
        ORDER BY scenario_id
        """
    ).fetchall()

    base_date = datetime(2026, 2, 10, 9, 0, 0)
    for cycle in range(2):
        for scenario in scenarios:
            for vehicle in vehicles:
                run_started = base_date + timedelta(days=cycle * 3 + scenario["scenario_id"], hours=vehicle["vehicle_id"])
                avg_speed = generate_average_speed(scenario, rng)
                cursor = conn.execute(
                    """
                    INSERT INTO Test_Runs (vehicle_id, scenario_id, test_date, avg_speed)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        vehicle["vehicle_id"],
                        scenario["scenario_id"],
                        run_started.isoformat(timespec="seconds"),
                        round(avg_speed, 2),
                    ),
                )
                run_id = cursor.lastrowid
                seed_frames_and_detections(conn, run_id, scenario, vehicle["accuracy_score"], rng)

    detection_count = conn.execute("SELECT COUNT(*) AS count FROM Detections").fetchone()["count"]
    if detection_count < 100:
        raise ValueError("Seed generation did not create the required 100+ detection samples.")


def seed_frames_and_detections(
    conn: sqlite3.Connection,
    run_id: int,
    scenario: sqlite3.Row,
    model_accuracy: float,
    rng: random.Random,
) -> None:
    frame_total = 10
    start_time = datetime(2026, 2, 10, 10, 0, 0)

    for frame_index in range(frame_total):
        frame_stamp = start_time + timedelta(seconds=frame_index * 2)
        image_path = f"seed/run_{run_id}/frame_{frame_index + 1:02d}.jpg"
        frame_cursor = conn.execute(
            """
            INSERT INTO Frames (run_id, timestamp, image_path)
            VALUES (?, ?, ?)
            """,
            (run_id, frame_stamp.isoformat(timespec="seconds"), image_path),
        )
        frame_id = frame_cursor.lastrowid

        detection_total = 2 if frame_index % 2 == 0 else 1
        for _ in range(detection_total):
            confidence = generate_confidence(model_accuracy, scenario, rng, historical=True)
            if scenario["weather_type"] in {"Rain", "Fog"} and rng.random() < 0.25:
                confidence = clamp(confidence - rng.uniform(0.14, 0.24), 0.42, 0.92)
            conn.execute(
                """
                INSERT INTO Detections (frame_id, object_type, confidence_score)
                VALUES (?, ?, ?)
                """,
                (
                    frame_id,
                    rng.choice(OBJECT_TYPES),
                    round(confidence, 3),
                ),
            )


def generate_confidence(
    model_accuracy: float,
    scenario: sqlite3.Row | dict[str, Any],
    rng: random.Random,
    historical: bool = False,
) -> float:
    baseline = model_accuracy + WEATHER_PENALTY[scenario["weather_type"]] + DIFFICULTY_PENALTY[scenario["difficulty"]]
    noise = rng.uniform(-0.08, 0.08 if historical else 0.12)
    return clamp(baseline + noise, 0.35, 0.985)


def generate_average_speed(scenario: sqlite3.Row | dict[str, Any], rng: random.Random) -> float:
    min_speed, max_speed = ROAD_SPEED[scenario["road_type"]]
    speed_shift = -6 if scenario["weather_type"] == "Fog" else -3 if scenario["weather_type"] == "Rain" else 0
    difficulty_shift = {"Low": 0, "Medium": -2, "High": -4, "Critical": -7}[scenario["difficulty"]]
    return clamp(rng.uniform(min_speed, max_speed) + speed_shift + difficulty_shift, 18, 84)


def fetch_scenarios(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    scenarios = conn.execute(
        """
        SELECT scenario_id, weather_type, light_level, road_type, difficulty
        FROM Scenarios
        ORDER BY scenario_id
        """
    ).fetchall()
    return [
        {
            **dict(row),
            "label": scenario_label(row),
            "overlay": row["weather_type"].lower(),
        }
        for row in scenarios
    ]


def fetch_vehicles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            v.vehicle_id,
            v.make_model,
            v.status,
            m.version_name,
            m.architecture,
            m.accuracy_score
        FROM Vehicles v
        JOIN AI_Models m ON m.model_id = v.ai_model_id
        ORDER BY v.vehicle_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_model_trend(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT model_id, version_name, architecture, accuracy_score
        FROM AI_Models
        ORDER BY model_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def average_confidence_for_model_and_scenario(
    conn: sqlite3.Connection, version_name: str, scenario_id: int
) -> float | None:
    row = conn.execute(
        """
        SELECT ROUND(AVG(d.confidence_score), 3) AS avg_confidence
        FROM AI_Models m
        JOIN Vehicles v ON v.ai_model_id = m.model_id
        JOIN Test_Runs tr ON tr.vehicle_id = v.vehicle_id
        JOIN Frames f ON f.run_id = tr.run_id
        JOIN Detections d ON d.frame_id = f.frame_id
        WHERE m.version_name = ? AND tr.scenario_id = ?
        """,
        (version_name, scenario_id),
    ).fetchone()
    return row["avg_confidence"]


def get_low_confidence_edge_frames(
    conn: sqlite3.Connection, threshold: float = 0.70
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            f.frame_id,
            f.timestamp,
            f.image_path,
            tr.run_id,
            s.weather_type,
            s.light_level,
            s.road_type,
            v.make_model,
            m.version_name,
            ROUND(MIN(d.confidence_score), 3) AS lowest_confidence
        FROM Frames f
        JOIN Test_Runs tr ON tr.run_id = f.run_id
        JOIN Scenarios s ON s.scenario_id = tr.scenario_id
        JOIN Vehicles v ON v.vehicle_id = tr.vehicle_id
        JOIN AI_Models m ON m.model_id = v.ai_model_id
        JOIN Detections d ON d.frame_id = f.frame_id
        WHERE d.confidence_score < ?
          AND s.weather_type IN ('Rain', 'Fog')
        GROUP BY
            f.frame_id,
            f.timestamp,
            f.image_path,
            tr.run_id,
            s.weather_type,
            s.light_level,
            s.road_type,
            v.make_model,
            m.version_name
        ORDER BY lowest_confidence ASC, f.frame_id ASC
        """,
        (threshold,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_detection_palette(conn: sqlite3.Connection, scenario_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT d.object_type, ROUND(AVG(d.confidence_score), 3) AS confidence_score
        FROM Detections d
        JOIN Frames f ON f.frame_id = d.frame_id
        JOIN Test_Runs tr ON tr.run_id = f.run_id
        WHERE tr.scenario_id = ?
        GROUP BY d.object_type
        ORDER BY confidence_score DESC
        """,
        (scenario_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def run_benchmark_simulation(vehicle_id: int, scenario_id: int) -> dict[str, Any]:
    rng = random.Random()
    with get_connection() as conn:
        scenario = conn.execute(
            """
            SELECT scenario_id, weather_type, light_level, road_type, difficulty
            FROM Scenarios
            WHERE scenario_id = ?
            """,
            (scenario_id,),
        ).fetchone()
        vehicle = conn.execute(
            """
            SELECT
                v.vehicle_id,
                v.make_model,
                v.status,
                m.version_name,
                m.architecture,
                m.accuracy_score
            FROM Vehicles v
            JOIN AI_Models m ON m.model_id = v.ai_model_id
            WHERE v.vehicle_id = ?
            """,
            (vehicle_id,),
        ).fetchone()

        if scenario is None or vehicle is None:
            raise ValueError("Invalid scenario or vehicle selection.")

        run_started = datetime.now().replace(microsecond=0)
        run_cursor = conn.execute(
            """
            INSERT INTO Test_Runs (vehicle_id, scenario_id, test_date, avg_speed)
            VALUES (?, ?, ?, 0)
            """,
            (vehicle_id, scenario_id, run_started.isoformat(timespec="seconds")),
        )
        run_id = run_cursor.lastrowid

        palette = get_detection_palette(conn, scenario_id) or [
            {"object_type": "Pedestrian", "confidence_score": 0.81},
            {"object_type": "Car", "confidence_score": 0.88},
        ]

        frame_events: list[dict[str, Any]] = []
        pass_count = 0
        confidence_values: list[float] = []
        speeds: list[float] = []

        for index in range(10):
            pass_roll = rng.random()
            passed = pass_roll < vehicle["accuracy_score"]
            seed_pick = rng.choice(palette)
            raw_confidence = generate_confidence(vehicle["accuracy_score"], scenario, rng)
            calibrated_confidence = (raw_confidence + seed_pick["confidence_score"]) / 2
            if not passed:
                calibrated_confidence -= rng.uniform(0.12, 0.22)
            confidence = round(clamp(calibrated_confidence, 0.38, 0.985), 3)

            object_type = seed_pick["object_type"]
            speed = round(generate_average_speed(scenario, rng), 2)
            speeds.append(speed)
            confidence_values.append(confidence)
            pass_count += int(passed)

            frame_time = run_started + timedelta(seconds=index * 2)
            frame_cursor = conn.execute(
                """
                INSERT INTO Frames (run_id, timestamp, image_path)
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    frame_time.isoformat(timespec="seconds"),
                    f"live/run_{run_id}/frame_{index + 1:02d}.jpg",
                ),
            )
            frame_id = frame_cursor.lastrowid

            detections_to_store = [
                (frame_id, object_type, confidence),
                (
                    frame_id,
                    rng.choice(OBJECT_TYPES),
                    round(clamp(confidence + rng.uniform(-0.08, 0.06), 0.34, 0.99), 3),
                ),
            ]
            conn.executemany(
                """
                INSERT INTO Detections (frame_id, object_type, confidence_score)
                VALUES (?, ?, ?)
                """,
                detections_to_store,
            )

            frame_events.append(
                {
                    "frame_number": index + 1,
                    "frame_id": frame_id,
                    "timestamp": frame_time.isoformat(timespec="seconds"),
                    "status": "PASS" if passed else "FAIL",
                    "object_type": object_type,
                    "confidence_score": confidence,
                    "probability_roll": round(pass_roll, 3),
                    "bounding_box": {
                        "left": rng.randint(10, 62),
                        "top": rng.randint(12, 56),
                        "width": rng.randint(18, 28),
                        "height": rng.randint(20, 28),
                    },
                    "speed": speed,
                }
            )

        avg_speed = round(mean(speeds), 2)
        conn.execute(
            "UPDATE Test_Runs SET avg_speed = ? WHERE run_id = ?",
            (avg_speed, run_id),
        )
        conn.commit()

        success_rate = round((pass_count / len(frame_events)) * 100, 1)
        average_confidence = round(mean(confidence_values), 3)

        return {
            "run_id": run_id,
            "scenario": {**dict(scenario), "label": scenario_label(scenario)},
            "vehicle": dict(vehicle),
            "frames": frame_events,
            "report": {
                "success_rate": success_rate,
                "passes": pass_count,
                "fails": len(frame_events) - pass_count,
                "average_confidence": average_confidence,
                "avg_speed": avg_speed,
                "model_accuracy": round(vehicle["accuracy_score"] * 100, 1),
            },
        }


@app.route("/")
def index() -> str:
    init_db()
    return render_template("index.html")


@app.route("/api/bootstrap")
def bootstrap() -> Any:
    init_db()
    with get_connection() as conn:
        scenarios = fetch_scenarios(conn)
        vehicles = fetch_vehicles(conn)
        models = fetch_model_trend(conn)
        default_scenario_id = scenarios[0]["scenario_id"]
        model_averages = [
            {
                "version_name": model["version_name"],
                "avg_confidence": average_confidence_for_model_and_scenario(
                    conn, model["version_name"], default_scenario_id
                ),
            }
            for model in models
        ]
        return jsonify(
            {
                "scenarios": scenarios,
                "vehicles": vehicles,
                "models": models,
                "edge_cases": get_low_confidence_edge_frames(conn)[:8],
                "model_averages": model_averages,
                "normalization_ok": run_normalization_check(conn),
            }
        )


@app.route("/api/analytics/avg-confidence")
def average_confidence_api() -> Any:
    init_db()
    version_name = request.args.get("version_name", "")
    scenario_id = request.args.get("scenario_id", type=int)
    if not version_name or scenario_id is None:
        return jsonify({"error": "version_name and scenario_id are required"}), 400

    with get_connection() as conn:
        average_confidence = average_confidence_for_model_and_scenario(conn, version_name, scenario_id)
        return jsonify(
            {
                "version_name": version_name,
                "scenario_id": scenario_id,
                "average_confidence": average_confidence,
            }
        )


@app.route("/api/edge-cases")
def edge_cases_api() -> Any:
    init_db()
    threshold = request.args.get("threshold", default=0.70, type=float)
    with get_connection() as conn:
        return jsonify({"frames": get_low_confidence_edge_frames(conn, threshold)})


@app.route("/api/run-benchmark", methods=["POST"])
def run_benchmark_api() -> Any:
    init_db()
    payload = request.get_json(silent=True) or {}
    scenario_id = payload.get("scenario_id")
    vehicle_id = payload.get("vehicle_id")

    if scenario_id is None or vehicle_id is None:
        return jsonify({"error": "scenario_id and vehicle_id are required"}), 400

    try:
        result = run_benchmark_simulation(int(vehicle_id), int(scenario_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
