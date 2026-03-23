import argparse
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "drivelearn.db"


def format_rows(cursor: sqlite3.Cursor, rows: list[sqlite3.Row]) -> str:
    if cursor.description is None:
        return "Query executed successfully."

    headers = [column[0] for column in cursor.description]
    table_rows = [headers] + [[str(row[index]) for index in range(len(headers))] for row in rows]
    widths = [max(len(row[index]) for row in table_rows) for index in range(len(headers))]

    def render(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    output = [render(headers), separator]
    output.extend(render(row) for row in table_rows[1:])
    output.append(f"\n{len(rows)} row(s).")
    return "\n".join(output)


def run_sql(sql: str) -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")

    try:
        cursor = connection.cursor()
        statements = [part.strip() for part in sql.split(";") if part.strip()]

        for index, statement in enumerate(statements, start=1):
            cursor.execute(statement)

            if cursor.description is not None:
                rows = cursor.fetchall()
                print(f"\nStatement {index}:")
                print(format_rows(cursor, rows))
            else:
                connection.commit()
                print(f"\nStatement {index}: Query executed successfully.")
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQLite queries against drivelearn.db")
    parser.add_argument("--sql", help="Inline SQL to execute")
    parser.add_argument("--file", help="Path to a .sql file to execute")
    args = parser.parse_args()

    if not args.sql and not args.file:
        parser.error("Provide either --sql or --file.")

    if args.sql and args.file:
        parser.error("Use only one of --sql or --file.")

    if args.file:
        sql = Path(args.file).read_text(encoding="utf-8")
    else:
        sql = args.sql

    run_sql(sql)


if __name__ == "__main__":
    main()
