# PostgreSQL Database Standards

**Domain:** Backend & Data Engineering  
**Skill:** High-Performance Relational Database Design  
**Context:** Designing, querying, and managing scalable PostgreSQL schemas for high-throughput applications involving large datasets and complex relationships.

---

## 1. Schema Design & Data Integrity
*Building a solid foundation that prevents data corruption and scales logically.*



### 1.1 Strict Typing & Constraints
* **Standard:** Use the most specific data type available.
    * **Use:** `TIMESTAMPTZ` (always store UTC), `UUID` (for distributed IDs), `TEXT` (over `VARCHAR(n)` unless a hard limit is strictly required), `JSONB` (for unstructured data that needs indexing).
    * **Avoid:** `MONEY` (use `DECIMAL`), `FLOAT` (for financial data), `TIMESTAMP` (without timezone).
* **Constraints:** Enforce data integrity at the database level, not just the application level.
    * Use `NOT NULL` by default.
    * Use `FOREIGN KEY` constraints for all relationships.
    * Use `CHECK` constraints for valid ranges (e.g., `percentage >= 0 AND percentage <= 100`).

### 1.2 Normalization Strategy
* **Standard:** Start with **3rd Normal Form (3NF)** to ensure data consistency and reduce redundancy.
* **Exception:** selective **Denormalization** is permitted *only* for proven read-heavy performance bottlenecks (e.g., storing a running `total_count` on a parent table to avoid expensive `COUNT(*)` queries), but must be kept in sync via Triggers.

### 1.3 Primary Keys
* **Standard:**
    * Use `BIGINT GENERATED ALWAYS AS IDENTITY` for internal, sequential efficiency (clustering).
    * Use `UUID` (v4 or v7) for external-facing IDs to prevent enumeration attacks and enable easy merging of distributed data.
    * **Anti-Pattern:** Using meaningful business data (like emails or usernames) as Primary Keys.

---

## 2. Indexing & Performance
*Ensuring queries remain fast as data grows from thousands to millions of rows.*



### 2.1 Indexing Strategy
* **Standard:**
    * **Foreign Keys:** Always index FK columns (Postgres does not do this automatically).
    * **B-Tree:** Default for equality and range queries (`=`, `<`, `>`).
    * **GIN:** Mandatory for `JSONB` containment queries (`@>`) and Full Text Search (`tsvector`).
    * **Partial Indexes:** Use `WHERE` clauses in indexes for localized optimization (e.g., `CREATE INDEX ... WHERE status = 'active'`) to reduce index size and maintenance overhead.
* **Anti-Pattern:** "Over-indexing" (creating an index for every column). This kills write performance (INSERT/UPDATE/DELETE).

### 2.2 Query Optimization
* **Standard:**
    * **Avoid `SELECT *`:** Fetch only required columns to reduce network I/O and serialization overhead.
    * **Avoid `N+1` Queries:** Use `JOIN`s or `WHERE ... IN (...)` instead of looping in code.
    * **SARGable Queries:** Ensure predicates allow index usage (e.g., `WHERE created_at > '2023-01-01'` works; `WHERE EXTRACT(YEAR FROM created_at) = 2023` scans the whole table).

### 2.3 EXPLAIN ANALYZE
* **Standard:** No complex query merges to `main` without an `EXPLAIN (ANALYZE, BUFFERS)` plan proving it uses indexes effectively (Seq Scans on large tables are forbidden).

---

## 3. Concurrency & Locking
*Handling high traffic without deadlocks or race conditions.*

### 3.1 Transaction Isolation
* **Standard:** Default to `READ COMMITTED`.
* **Requirement:** Use `SELECT ... FOR UPDATE` explicitly when reading data that you intend to modify within the same transaction to prevent race conditions.
* **Anti-Pattern:** Long-running transactions. They hold locks and prevent `VACUUM` from cleaning up dead tuples, causing table bloat.

### 3.2 Deadlock Avoidance
* **Standard:** Access tables in a consistent order across all transactions (e.g., always update `Users` then `Orders`, never the reverse).
* **Practice:** Keep transactions as short as possible. Do not make external API calls inside a database transaction.

---

## 4. Maintenance & Operations
*Keeping the database healthy over time.*



### 4.1 Vacuuming & Bloat
* **Standard:** Ensure `autovacuum` is enabled and tuned.
    * **Monitoring:** Alert on "Dead Tuple" percentage (> 5-10%).
    * **Action:** Regular `VACUUM ANALYZE` updates statistics so the Query Planner makes good decisions.

### 4.2 Migrations (Schema Changes)
* **Standard:**
    * Use a migration tool (e.g., Alembic, Flyway).
    * **Zero-Downtime:** Adding a column with a `DEFAULT` value on a huge table locks it. instead:
        1. Add column nullable (fast).
        2. Backfill data in batches.
        3. Add `NOT NULL` constraint.
    * **Concurrent Indexes:** Always use `CREATE INDEX CONCURRENTLY` in production to avoid locking the table against writes.

---

## 5. Security & Access
*Protecting data at the source.*

### 5.1 Role-Based Access Control (RBAC)
* **Standard:**
    * **App User:** Should only have `CONNECT`, `SELECT`, `INSERT`, `UPDATE`, `DELETE`. strictly **NO** `TRUNCATE`, `DROP`, or `ALTER`.
    * **Migration User:** Separate credentials for schema changes (CI/CD pipeline).
    * **Read-Only:** Specific user for analytics/reporting to prevent accidental writes.

### 5.2 Row-Level Security (RLS)
* **Standard:** For multi-tenant applications (SaaS), enable RLS on sensitive tables to enforce tenant isolation at the database engine level, acting as a final safety net if application logic fails.

---

## Checklist for Code Review
Before merging database changes:

- [ ] **Types:** Are correct data types used (`TIMESTAMPTZ`, `JSONB`)?
- [ ] **Constraints:** Are `NOT NULL` and `FOREIGN KEY` constraints applied?
- [ ] **Indexes:** Are Foreign Keys indexed? Is `CREATE INDEX CONCURRENTLY` used for existing tables?
- [ ] **Performance:** Is there an `EXPLAIN ANALYZE` plan for complex queries?
- [ ] **Safety:** Are dangerous operations (dropping columns/tables) handled with a distinct migration step?
- [ ] **Locking:** Are transactions short, with no external API calls inside?
- [ ] **Naming:** Do table/column names follow `snake_case` conventions?