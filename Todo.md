# MediCortex AI 2.0 — Technical Todo

---

## 🔴 High Priority

### Patient Database — Replace Simulated In-Memory Store with PostgreSQL

Currently `tools/patient_retriever_tools.py` uses a hardcoded `_SIMULATED_PATIENTS` dict. This needs to be replaced with a real queryable patient table backed by the existing PostgreSQL connection.

**What to build**:
- Add a `patients` table to `database/schema.sql` covering demographics, diagnoses (JSONB), medications (JSONB), allergies (JSONB), and vitals history (JSONB)
- Add a `Patient` SQLAlchemy model to `database/models.py`
- Seed the table with the existing 4 simulated patients via a migration script in `tools/migrate_db.py`
- Refactor `tools/patient_retriever_tools.py` to query the `patients` table via the existing async SQLAlchemy session from `database/connection.py` instead of the dict lookup

**Files to change**:
- `database/schema.sql` — add `patients` table DDL
- `database/models.py` — add `Patient` ORM model
- `database/init_db.py` — include `patients` table in schema init
- `tools/migrate_db.py` — seed script to insert the 4 existing demo patients
- `tools/patient_retriever_tools.py` — replace `_SIMULATED_PATIENTS` dict with async DB query
