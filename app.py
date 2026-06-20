"""
Auto Inventory Dashboard
-------------------------
A small Flask web app to manage a vehicle inventory (CRUD) and
visualize it on a dashboard (charts by brand, fuel type, price, year).

Run:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000
"""

import os
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "auto.db")

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the cars table if it doesn't exist, and seed sample data once."""
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            price REAL NOT NULL,
            fuel_type TEXT NOT NULL,
            mileage INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'Available',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS c FROM cars").fetchone()["c"]
    if count == 0:
        seed = [
            ("Toyota", "Camry", 2022, 28500, "Petrol", 15000, "Available"),
            ("Toyota", "Corolla", 2023, 23000, "Hybrid", 8000, "Available"),
            ("Honda", "Civic", 2021, 24500, "Petrol", 22000, "Sold"),
            ("Honda", "CR-V", 2023, 32000, "Hybrid", 5000, "Available"),
            ("Tesla", "Model 3", 2023, 41000, "Electric", 3000, "Available"),
            ("Tesla", "Model Y", 2022, 49500, "Electric", 12000, "Reserved"),
            ("Ford", "F-150", 2021, 38000, "Petrol", 30000, "Sold"),
            ("Ford", "Mustang", 2023, 45000, "Petrol", 2000, "Available"),
            ("BMW", "3 Series", 2022, 47000, "Diesel", 18000, "Available"),
            ("BMW", "X5", 2023, 68000, "Diesel", 6000, "Reserved"),
            ("Hyundai", "Elantra", 2021, 21000, "Petrol", 25000, "Sold"),
            ("Hyundai", "Ioniq 5", 2023, 44000, "Electric", 4000, "Available"),
        ]
        now = datetime.utcnow().isoformat()
        conn.executemany(
            """
            INSERT INTO cars (brand, model, year, price, fuel_type, mileage, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [s + (now,) for s in seed],
        )
        conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@app.route("/")
def dashboard():
    conn = get_db()
    cars = conn.execute("SELECT * FROM cars").fetchall()
    conn.close()

    total = len(cars)
    total_value = sum(c["price"] for c in cars)
    avg_price = total_value / total if total else 0
    available = sum(1 for c in cars if c["status"] == "Available")

    stats = {
        "total": total,
        "total_value": round(total_value, 2),
        "avg_price": round(avg_price, 2),
        "available": available,
    }
    return render_template("dashboard.html", stats=stats)


@app.route("/api/chart-data")
def chart_data():
    """Aggregated data for the dashboard charts, consumed via fetch() in JS."""
    conn = get_db()
    cars = conn.execute("SELECT * FROM cars").fetchall()
    conn.close()

    by_brand = {}
    by_fuel = {}
    by_status = {}
    by_year = {}

    for c in cars:
        by_brand[c["brand"]] = by_brand.get(c["brand"], 0) + 1
        by_fuel[c["fuel_type"]] = by_fuel.get(c["fuel_type"], 0) + 1
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        by_year[c["year"]] = by_year.get(c["year"], 0) + c["price"]

    return jsonify(
        {
            "by_brand": by_brand,
            "by_fuel": by_fuel,
            "by_status": by_status,
            "by_year": dict(sorted(by_year.items())),
        }
    )


# --------------------------------------------------------------------------- #
# CRUD: list / create / edit / delete
# --------------------------------------------------------------------------- #
@app.route("/cars")
def list_cars():
    conn = get_db()
    q = request.args.get("q", "").strip()
    if q:
        cars = conn.execute(
            "SELECT * FROM cars WHERE brand LIKE ? OR model LIKE ? ORDER BY id DESC",
            (f"%{q}%", f"%{q}%"),
        ).fetchall()
    else:
        cars = conn.execute("SELECT * FROM cars ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("cars_list.html", cars=cars, q=q)


@app.route("/cars/new", methods=["GET", "POST"])
def new_car():
    if request.method == "POST":
        data = _extract_form()
        error = _validate(data)
        if error:
            flash(error, "error")
            return render_template("car_form.html", car=data, mode="new")

        conn = get_db()
        conn.execute(
            """
            INSERT INTO cars (brand, model, year, price, fuel_type, mileage, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["brand"],
                data["model"],
                data["year"],
                data["price"],
                data["fuel_type"],
                data["mileage"],
                data["status"],
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        flash(f"Added {data['brand']} {data['model']}.", "success")
        return redirect(url_for("list_cars"))

    return render_template("car_form.html", car=None, mode="new")


@app.route("/cars/<int:car_id>/edit", methods=["GET", "POST"])
def edit_car(car_id):
    conn = get_db()
    car = conn.execute("SELECT * FROM cars WHERE id = ?", (car_id,)).fetchone()
    if car is None:
        conn.close()
        flash("Car not found.", "error")
        return redirect(url_for("list_cars"))

    if request.method == "POST":
        data = _extract_form()
        error = _validate(data)
        if error:
            flash(error, "error")
            conn.close()
            return render_template("car_form.html", car={**data, "id": car_id}, mode="edit")

        conn.execute(
            """
            UPDATE cars
            SET brand = ?, model = ?, year = ?, price = ?, fuel_type = ?, mileage = ?, status = ?
            WHERE id = ?
            """,
            (
                data["brand"],
                data["model"],
                data["year"],
                data["price"],
                data["fuel_type"],
                data["mileage"],
                data["status"],
                car_id,
            ),
        )
        conn.commit()
        conn.close()
        flash(f"Updated {data['brand']} {data['model']}.", "success")
        return redirect(url_for("list_cars"))

    conn.close()
    return render_template("car_form.html", car=car, mode="edit")


@app.route("/cars/<int:car_id>/delete", methods=["POST"])
def delete_car(car_id):
    conn = get_db()
    car = conn.execute("SELECT * FROM cars WHERE id = ?", (car_id,)).fetchone()
    if car:
        conn.execute("DELETE FROM cars WHERE id = ?", (car_id,))
        conn.commit()
        flash(f"Deleted {car['brand']} {car['model']}.", "success")
    conn.close()
    return redirect(url_for("list_cars"))


# --------------------------------------------------------------------------- #
# Form helpers
# --------------------------------------------------------------------------- #
def _extract_form():
    return {
        "brand": request.form.get("brand", "").strip(),
        "model": request.form.get("model", "").strip(),
        "year": request.form.get("year", "").strip(),
        "price": request.form.get("price", "").strip(),
        "fuel_type": request.form.get("fuel_type", "Petrol"),
        "mileage": request.form.get("mileage", "0").strip(),
        "status": request.form.get("status", "Available"),
    }


def _validate(data):
    if not data["brand"] or not data["model"]:
        return "Brand and model are required."
    try:
        data["year"] = int(data["year"])
        if data["year"] < 1900 or data["year"] > 2100:
            return "Year must be a realistic number."
    except (ValueError, TypeError):
        return "Year must be a whole number."
    try:
        data["price"] = float(data["price"])
        if data["price"] < 0:
            return "Price can't be negative."
    except (ValueError, TypeError):
        return "Price must be a number."
    try:
        data["mileage"] = int(data["mileage"] or 0)
        if data["mileage"] < 0:
            return "Mileage can't be negative."
    except (ValueError, TypeError):
        return "Mileage must be a whole number."
    if data["status"] not in ("Available", "Reserved", "Sold"):
        return "Invalid status."
    if data["fuel_type"] not in ("Petrol", "Diesel", "Hybrid", "Electric"):
        return "Invalid fuel type."
    return None


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)