# FastAPI Development Standards

**Domain:** Backend Engineering  
**Skill:** High-Performance FastAPI Development  
**Context:** Building scalable services with multiple endpoints, complex database schemas (SQL/NoSQL), and high-volume data processing.

---

## 1. Concurrency & Event Loop Management
*The ability to write non-blocking code is the single most critical skill for high-performance FastAPI applications.*

### 1.1 `async def` vs. `def`
* **Standard:**
    * **Use `async def`:** Only when the function uses `await` for non-blocking I/O (e.g., querying DB with `asyncpg`, calling external APIs with `httpx`).
    * **Use `def`:** For any blocking I/O operations (e.g., `requests`, standard `open()`, synchronous DB drivers). FastAPI automatically runs these in a separate thread pool to prevent blocking the main event loop.
* **Anti-Pattern:** Using `time.sleep()` or synchronous DB calls inside an `async def` function. This freezes the entire application for all users.

### 1.2 Async-Native Libraries
* **Standard:** Prioritize libraries that support Python’s `await` syntax to maximize throughput.
    * **Network:** Use `httpx` (async) instead of `requests`.
    * **Database:** Use `Motor` (MongoDB) or `SQLAlchemy` 1.4+ (Async Mode) / `asyncpg` (PostgreSQL).
    * **Utilities:** Use `asyncio.sleep()` instead of `time.sleep()`.

---

## 2. Database & Data Architecture
*Strategies for handling multiple related tables and "huge data" without degrading performance.*

### 2.1 Connection Pooling & Lifespan
* **Standard:** Initialize database connection pools **once** during the application startup using the `lifespan` context manager.
* **Implementation:** Store the pool in `app.state` or a global dependency. Never create a new connection per request; borrow from the pool using `async with`.
* **Migration:** Move away from deprecated `@app.on_event("startup")` handlers.

### 2.2 Database Dependencies (DI)
* **Standard:** Use FastAPI's Dependency Injection (`Depends`) for database access and validation that requires DB lookups (e.g., `get_current_user`, `verify_ownership`).
* **Benefit:** Dependencies are cached per request, ensuring that if multiple endpoints or sub-dependencies need the user/DB, the query runs only once.

### 2.3 Global Data Encoding
* **Standard:** Configure global encoders or custom Pydantic `BaseConfig` to handle serialization of complex types automatically.
* **Use Case:**
    * Convert MongoDB `ObjectId` to string.
    * Format `datetime` objects to ISO strings.
    * Handle `Decimal` types for financial data.
    * Map `snake_case` (Python) to `camelCase` (JSON frontend) using alias generators.

---

## 3. Computational Workloads & Background Processing
*Handling data-heavy operations without blocking HTTP responses.*

### 3.1 Heavy Computation (CPU-Bound)
* **Standard:** Do **not** perform heavy calculations (e.g., video processing, heavy ML inference > 100ms) inside endpoints.
* **Architecture:**
    * **Lightweight:** Run small tasks in `def` endpoints (thread pool).
    * **Heavy:** Offload to a dedicated worker queue (Celery + RabbitMQ/Redis).
    * **ML Serving:** Use dedicated inference servers (Triton, TensorFlow Serving) and query them via API.

### 3.2 Fire-and-Forget Tasks
* **Standard:** Use FastAPI `BackgroundTasks` for non-critical operations that shouldn't block the user (e.g., sending confirmation emails, logging metrics).
* **Constraint:** For tasks requiring guaranteed delivery or retries, use a persistent queue (Celery) instead of `BackgroundTasks`.

---

## 4. Code Quality & Validation Standards
*Ensuring maintainability and robust error handling.*

### 4.1 Pydantic First
* **Standard:** Push **all** validation logic into Pydantic models. Avoid manual `if/else` validation checks inside route handlers.
* **Benefit:** Auto-generated documentation, consistent error responses, and centralized validation logic.

### 4.2 Response Model Automation
* **Standard:** Always define `response_model` in the route decorator.
* **Practice:** Return raw ORM objects or dictionaries from your function. Let FastAPI handle the conversion to JSON and filtering of sensitive data based on the Pydantic model.

### 4.3 Configuration Management
* **Standard:** Use `pydantic-settings` to manage environment variables.
* **Practice:**
    * Never hardcode secrets.
    * Use `.env` files for local development (git-ignored).
    * Fail fast: The app should crash on startup if required variables (DB_URL, API_KEY) are missing.

---

## 5. Production Operations (Ops)
*Requirements for deploying to the Antigravity production environment.*

### 5.1 Structured Logging
* **Standard:** **Forbidden** to use `print()` statements.
* **Requirement:** Use structured logging libraries (`structlog` or `loguru`).
* **Format:** Logs must be JSON formatted and include context identifiers (`request_id`, `user_id`, `trace_id`) to allow debugging across distributed systems.

### 5.2 Deployment Configuration
* **Process Manager:** Run the application using **Gunicorn** managing **Uvicorn workers** (`uvicorn.workers.UvicornWorker`).
* **Performance:**
    * Enable `uvloop` (usually automatic with standard Uvicorn installation) for event loop performance.
    * **Worker Count Formula:** `(2 x CPU_Cores) + 1` (Benchmark to fine-tune).
* **Security:** explicitly set `docs_url=None` and `redoc_url=None` in production environments to prevent schema leakage.

---

## Checklist for Code Review
Before merging to `main`:

- [ ] Are blocking operations wrapped in `def` or offloaded?
- [ ] Is the database connection pooled and managed via `lifespan`?
- [ ] Are complex validations handled in Pydantic models?
- [ ] Are `print()` statements replaced with structured logs?
- [ ] Is `response_model` used for all endpoints?
- [ ] Are secrets managed via `pydantic-settings`?