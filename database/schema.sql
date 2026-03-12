-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop tables if they exist (Be careful in production!)
DROP TABLE IF EXISTS chat_messages;
DROP TABLE IF EXISTS chat_sessions;

-- Create chat_sessions table
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create chat_messages table
CREATE TABLE chat_messages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    attachments JSONB DEFAULT '[]'::jsonb,
    thinking JSONB DEFAULT '[]'::jsonb,
    message_metadata JSONB DEFAULT '{}'::jsonb
);

-- Create Indexes
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_attachments ON chat_messages USING GIN (attachments);

-- ── Patients table (Synthea-compatible schema) ──────────────────────────────
-- Uses CREATE TABLE IF NOT EXISTS so init_db runs are idempotent and never
-- wipe seeded patient data.
CREATE TABLE IF NOT EXISTS patients (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      TEXT        UNIQUE NOT NULL,
    full_name       TEXT        NOT NULL,
    date_of_birth   DATE,
    age             INTEGER,
    sex             TEXT,
    blood_type      TEXT,
    race            TEXT,
    ethnicity       TEXT,
    marital_status  TEXT,
    address         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    diagnoses       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    medications     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    allergies       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    vitals_history  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    last_visit      DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patients_patient_id   ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_full_name    ON patients(lower(full_name));
CREATE INDEX IF NOT EXISTS idx_patients_diagnoses    ON patients USING GIN (diagnoses);
CREATE INDEX IF NOT EXISTS idx_patients_medications  ON patients USING GIN (medications);
