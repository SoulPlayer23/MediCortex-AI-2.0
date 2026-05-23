"""
OBS-1 — Observability dashboard API.

Reads message_metadata JSONB from chat_messages to surface per-node latency,
retrieval stats, judge token usage, and request timelines.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sa_text

from database.connection import get_db

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary")
async def dashboard_summary(db: AsyncSession = Depends(get_db)):
    """Aggregate stats across all assistant messages."""
    row = (await db.execute(sa_text("""
        SELECT
            COUNT(*) FILTER (WHERE role = 'assistant') AS total_responses,
            AVG((message_metadata->>'request_elapsed_ms')::float)
                FILTER (WHERE message_metadata->>'request_elapsed_ms' IS NOT NULL)
                AS avg_request_ms,
            AVG((message_metadata->>'judge_score')::float)
                FILTER (WHERE message_metadata->>'judge_score' IS NOT NULL)
                AS avg_judge_score,
            COUNT(*) FILTER (
                WHERE role = 'assistant'
                AND message_metadata->>'judge_score' IS NOT NULL
            ) AS judged_count
        FROM chat_messages
    """))).one()

    return {
        "total_responses": row.total_responses or 0,
        "avg_request_ms": round(row.avg_request_ms or 0, 1),
        "avg_judge_score": round(row.avg_judge_score or 0, 2),
        "judged_count": row.judged_count or 0,
    }


@router.get("/dashboard/agents")
async def dashboard_agents(db: AsyncSession = Depends(get_db)):
    """Per-agent usage counts from agents_used metadata arrays."""
    rows = (await db.execute(sa_text("""
        SELECT
            agent,
            COUNT(*) AS usage_count
        FROM chat_messages,
             jsonb_array_elements_text(
                 COALESCE(
                     NULLIF(message_metadata->'agents_used', 'null'::jsonb),
                     '[]'::jsonb
                 )
             ) AS agent
        WHERE role = 'assistant'
          AND jsonb_typeof(COALESCE(message_metadata->'agents_used', '[]'::jsonb)) = 'array'
        GROUP BY agent
        ORDER BY usage_count DESC
    """))).all()

    return [{"agent": r.agent, "usage_count": r.usage_count} for r in rows]


@router.get("/dashboard/node-timings")
async def dashboard_node_timings(db: AsyncSession = Depends(get_db)):
    """Average latency per graph node across all requests."""
    rows = (await db.execute(sa_text("""
        SELECT
            node_key AS node,
            AVG(node_ms::float) AS avg_ms,
            MIN(node_ms::float) AS min_ms,
            MAX(node_ms::float) AS max_ms,
            COUNT(*) AS sample_count
        FROM (
            SELECT
                kv.key AS node_key,
                kv.value AS node_ms
            FROM chat_messages,
                 jsonb_each_text(
                     COALESCE(message_metadata->'node_timings', '{}'::jsonb)
                 ) AS kv(key, value)
            WHERE role = 'assistant'
              AND message_metadata->'node_timings' IS NOT NULL
        ) sub
        GROUP BY node_key
        ORDER BY avg_ms DESC
    """))).all()

    return [
        {
            "node": r.node,
            "avg_ms": round(r.avg_ms or 0, 1),
            "min_ms": round(r.min_ms or 0, 1),
            "max_ms": round(r.max_ms or 0, 1),
            "sample_count": r.sample_count,
        }
        for r in rows
    ]


@router.get("/dashboard/requests")
async def dashboard_requests(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Recent request timeline — one row per assistant response."""
    rows = (await db.execute(sa_text("""
        SELECT
            id,
            session_id,
            timestamp,
            (message_metadata->>'request_elapsed_ms')::float AS request_elapsed_ms,
            (message_metadata->>'judge_score')::float AS judge_score,
            message_metadata->'agents_used' AS agents_used,
            message_metadata->'node_timings' AS node_timings,
            message_metadata->'retrieval' AS retrieval,
            message_metadata->'token_usage' AS token_usage
        FROM chat_messages
        WHERE role = 'assistant'
        ORDER BY timestamp DESC
        LIMIT :limit
    """), {"limit": min(limit, 200)})).all()

    return [
        {
            "id": r.id,
            "session_id": str(r.session_id),
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "request_elapsed_ms": r.request_elapsed_ms,
            "judge_score": r.judge_score,
            "agents_used": r.agents_used or [],
            "node_timings": r.node_timings or {},
            "retrieval": r.retrieval,
            "token_usage": r.token_usage,
        }
        for r in rows
    ]
