# Auto Inventory Dashboard

A Flask web app to manage a vehicle inventory: full CRUD (add/edit/delete/search)
plus a dashboard with live charts (by brand, fuel type, status, and value by year).

## Setup

```bash
cd auto-dashboard
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

A SQLite database (`auto.db`) is created automatically on first run, seeded
with 12 sample vehicles so the dashboard isn't empty.

## Structure

```
auto-dashboard/
├── app.py                  # All routes + SQLite logic
├── requirements.txt
├── templates/
│   ├── base.html           # Shared layout/nav
│   ├── dashboard.html      # Stat cards + 4 charts
│   ├── cars_list.html      # Inventory table + search
│   └── car_form.html       # Add/edit form (shared)
└── static/
    ├── css/style.css       # Theming
    └── js/dashboard.js     # Chart.js rendering
```

## Pages

- `/` — Dashboard: total vehicles, available count, total/average value,
  and 4 charts (bar, two doughnuts, line) fed by `/api/chart-data`.
- `/cars` — Inventory list with search by brand/model.
- `/cars/new` — Add a vehicle.
- `/cars/<id>/edit` — Edit a vehicle.
- `/cars/<id>/delete` (POST) — Delete a vehicle.

## Notes

- Storage is SQLite (`auto.db`), zero external services needed.
- Charts use Chart.js loaded from a CDN.
- To start fresh, just delete `auto.db` — it'll reseed on next run.
- This is a dev setup (`debug=True`, dev `secret_key`) — change both before
  deploying anywhere public.
