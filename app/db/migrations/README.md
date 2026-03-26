Alembic migrations directory placeholder.

Initialize with:

- `alembic init app/db/migrations`
- configure `env.py` to import `app.db.session.Base`
- generate revision: `alembic revision --autogenerate -m "init"`
