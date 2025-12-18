"""Database helpers for student roster uploads."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor


def _table_exists(table_name: str) -> bool:
    query = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %(table_name)s
        LIMIT 1
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"table_name": table_name})
        return cur.fetchone() is not None


def fetch_existing_enrollments(admin_id: int, enrollments: Iterable[str]) -> List[str]:
    """Return enrollment numbers that already exist for the admin."""

    enrollment_list = list(enrollments)
    if not enrollment_list:
        return []

    query = (
        "SELECT enrollment_number FROM student_roster_entries "
        "WHERE admin_id = %(admin_id)s AND enrollment_number = ANY(%(enrollments)s)"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id, "enrollments": enrollment_list})
        rows = cur.fetchall()

    return [row["enrollment_number"] for row in rows]


def insert_roster_entries(admin_id: int, entries: List[Dict[str, Any]]) -> None:
    """Insert roster entries for an admin."""

    if not entries:
        return

    insert_sql = (
        "INSERT INTO student_roster_entries "
        "(admin_id, enrollment_number, first_name,middle_name, last_name, std, division, auto_password, assigned_member_id) "
        "VALUES (%(admin_id)s, %(enrollment_number)s, %(first_name)s, %(middle_name)s, %(last_name)s, %(std)s, %(division)s, %(auto_password)s, %(assigned_member_id)s)"
    )

    payload = [
        {
            "admin_id": admin_id,
            "enrollment_number": entry["enrollment_number"],
            "first_name": entry["first_name"],
            "middle_name": entry.get("middle_name"),
            "last_name": entry.get("last_name"),
            "std": entry["std"],
            "division": entry.get("division"),
            "auto_password": entry["auto_password"],
            "assigned_member_id": entry.get("assigned_member_id"),
        }
        for entry in entries
    ]

    with get_pg_cursor(dict_rows=False) as cur:
        cur.executemany(insert_sql, payload)


def update_roster_entry(
    admin_id: int,
    enrollment_number: str,
    *,
    member_id: Optional[int] = None,
    **fields: Any,
) -> Optional[Dict[str, Any]]:
    """Update a roster entry for the admin and return the updated row."""

    if not fields:
        return None

    allowed_columns = {"first_name","middle_name", "last_name", "std", "division", "auto_password"}
    invalid = set(fields.keys()) - allowed_columns
    if invalid:
        raise ValueError(f"Invalid roster columns: {', '.join(sorted(invalid))}")

    assignments = [f"{column} = %({column})s" for column in fields.keys()]
    params: Dict[str, Any] = {"admin_id": admin_id, "enrollment_number": enrollment_number, **fields}
    conditions = ["admin_id = %(admin_id)s", "enrollment_number = %(enrollment_number)s"]
    if member_id is not None:
        params["member_id"] = member_id
        conditions.append("assigned_member_id = %(member_id)s")

    query = (
        "UPDATE student_roster_entries "
        f"SET {', '.join(assignments)} "
        "WHERE " + " AND ".join(conditions) + " "
        "RETURNING enrollment_number, first_name, last_name, std, division, auto_password, created_at"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def delete_roster_entry(admin_id: int, enrollment_number: str, *, member_id: Optional[int] = None) -> bool:
    """Delete a single roster entry for the admin."""

    conditions = ["admin_id = %(admin_id)s", "enrollment_number = %(enrollment_number)s"]
    params: Dict[str, Any] = {"admin_id": admin_id, "enrollment_number": enrollment_number}
    if member_id is not None:
        params["member_id"] = member_id
        conditions.append("assigned_member_id = %(member_id)s")

    query = (
        "DELETE FROM student_roster_entries "
        "WHERE " + " AND ".join(conditions) + " "
        "RETURNING enrollment_number"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone() is not None


def count_roster_students(admin_id: int, *, member_id: Optional[int] = None) -> int:
    """Return the total number of rostered students for an admin (optionally per member)."""

    conditions = ["admin_id = %(admin_id)s"]
    params: Dict[str, Any] = {"admin_id": admin_id}

    if member_id is not None:
        conditions.append("assigned_member_id = %(member_id)s")
        params["member_id"] = member_id

    query = (
        "SELECT COUNT(*) FROM student_roster_entries "
        "WHERE " + " AND ".join(conditions)
    )

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
        (count,) = cur.fetchone()
    return int(count)


def fetch_roster_entries(admin_id: int, *, member_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return stored roster entries for an admin ordered by latest."""

    params: Dict[str, Any] = {"admin_id": admin_id}
    conditions = ["admin_id = %(admin_id)s"]
    if member_id is not None:
        params["member_id"] = member_id
        conditions.append("assigned_member_id = %(member_id)s")

    query = (
        "SELECT enrollment_number, first_name, middle_name, last_name, std, division, auto_password, created_at "
        "FROM student_roster_entries "
        "WHERE " + " AND ".join(conditions) + " "
        "ORDER BY created_at DESC"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def fetch_roster_entry_by_enrollment(
    admin_id: int,
    enrollment_number: str,
    *,
    member_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return a single roster entry for the admin by enrollment number."""

    conditions = ["admin_id = %(admin_id)s", "LOWER(enrollment_number) = LOWER(%(enrollment_number)s)"]
    params: Dict[str, Any] = {"admin_id": admin_id, "enrollment_number": enrollment_number}
    if member_id is not None:
        params["member_id"] = member_id
        conditions.append("assigned_member_id = %(member_id)s")

    query = (
        "SELECT enrollment_number, first_name, last_name, std, division, auto_password, created_at "
        "FROM student_roster_entries "
        "WHERE " + " AND ".join(conditions) + " "
        "LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def fetch_student_profiles(
    admin_id: int,
    *,
    class_filter: Optional[str] = None,
    division_filter: Optional[str] = None,
    enrollment_filter: Optional[str] = None,
    member_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return roster entries enriched with profile details for the admin."""

    purchases_available = _table_exists("purchases")
    params: Dict[str, Any] = {"admin_id": admin_id}
    conditions = ["r.admin_id = %(admin_id)s"]

    if member_id is not None:
        params["member_id"] = member_id
        conditions.append("r.assigned_member_id = %(member_id)s")

    if enrollment_filter:
        params["enrollment_filter"] = enrollment_filter
        conditions.append("LOWER(r.enrollment_number) = LOWER(%(enrollment_filter)s)")

    if class_filter:
        params["class_filter"] = class_filter
        conditions.append("(r.std = %(class_filter)s OR p.class_stream = %(class_filter)s)")

    if division_filter:
        params["division_filter"] = division_filter
        conditions.append("(r.division = %(division_filter)s OR p.division = %(division_filter)s)")

    where_clause = " AND ".join(conditions)

    if purchases_available:
        query = """
            SELECT
                r.enrollment_number,
                r.first_name        AS roster_first_name,
                r.middle_name        AS roster_middle_name,
                r.last_name         AS roster_last_name,
                r.std,
                r.division          AS roster_division,
                r.auto_password,
                r.created_at        AS roster_created_at,
                p.id                AS profile_id,
                p.first_name        AS profile_first_name,
                p.middle_name        AS roster_middle_name,
                p.last_name         AS roster_last_name,
                p.class_stream,
                p.division          AS profile_division,
                p.class_head,
                p.mobile_number,
                p.parents_number,
                p.email,
                p.photo_path,
                p.created_at        AS profile_created_at,
                (p.id IS NOT NULL)  AS profile_complete,
                COALESCE(
                    json_agg(
                        jsonb_build_object(
                            'transaction_id', pur.transaction_id,
                            'date', pur.date,
                            'time', pur.time,
                            'payment_method', pur.payment_method,
                            'transaction_type', pur.transaction_type,
                            'description', pur.description,
                            'amount', pur.amount,
                            'status', pur.status
                        )
                        ORDER BY pur.date DESC, pur.time DESC
                    ) FILTER (WHERE pur.transaction_id IS NOT NULL),
                    '[]'::json
                )                 AS purchases
            FROM student_roster_entries r
            LEFT JOIN student_profiles p ON p.enrollment_number = r.enrollment_number
            LEFT JOIN students s ON s.enrollment_number = r.enrollment_number
            LEFT JOIN purchases pur ON pur.sid = s.sid
            WHERE {where_clause}
            GROUP BY
                r.id,
                r.enrollment_number,
                r.first_name,
                r.middle_name,
                r.last_name,
                r.std,
                r.division,
                r.auto_password,
                r.created_at,
                p.id,
                p.first_name,
                p.middle_name,
                p.last_name,
                p.class_stream,
                p.division,
                p.class_head,
                p.mobile_number,
                p.parents_number,
                p.email,
                p.photo_path,
                p.created_at
            ORDER BY r.created_at DESC
        """
    else:
        query = """
            SELECT
                r.enrollment_number,
                r.first_name        AS roster_first_name,
                r.middle_name       AS roster_middle_name,
                r.last_name         AS roster_last_name,
                r.std,
                r.division          AS roster_division,
                r.auto_password,
                r.created_at        AS roster_created_at,
                p.id                AS profile_id,
                p.first_name        AS profile_first_name,
                p.middle_name       AS roster_middle_name,
                p.last_name         AS roster_last_name,
                p.class_stream,
                p.division          AS profile_division,
                p.class_head,
                p.mobile_number,
                p.parents_number,
                p.email,
                p.photo_path,
                p.created_at        AS profile_created_at,
                (p.id IS NOT NULL)  AS profile_complete,
                '[]'::json          AS purchases
            FROM student_roster_entries r
            LEFT JOIN student_profiles p ON p.enrollment_number = r.enrollment_number
            WHERE {where_clause}
        """

    with get_pg_cursor() as cur:
        cur.execute(query.format(where_clause=where_clause), params)
        rows = cur.fetchall()

    for row in rows:
        purchases = row.get("purchases")
        if isinstance(purchases, str):
            try:
                row["purchases"] = json.loads(purchases)
            except json.JSONDecodeError:
                row["purchases"] = []
        elif purchases is None:
            row["purchases"] = []

    return rows


def fetch_class_subjects(
    admin_id: int,
    *,
    class_filter: Optional[str],
    division_filter: Optional[str] = None,
    member_id: Optional[int] = None,
) -> List[str]:
    """Return distinct subjects for the given class using chapter materials data."""

    if not class_filter:
        return []

    base_condition = """
        admin_id = %(admin_id)s
          AND TRIM(subject) <> ''
          AND subject IS NOT NULL
          AND (
                std = %(class_filter)s
             OR LOWER(TRIM(std)) = LOWER(TRIM(%(class_filter)s))
             OR LOWER(std) LIKE LOWER(%(class_like)s)
          )
    """

    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "class_filter": class_filter,
        "class_like": f"%{class_filter}%",
    }

    member_clause = ""
    if member_id is not None:
        params["member_id"] = member_id
        member_clause = """
          AND EXISTS (
                SELECT 1
                FROM student_roster_entries sre
                WHERE sre.admin_id = chapter_materials.admin_id
                  AND sre.std = chapter_materials.std
                  AND sre.assigned_member_id = %(member_id)s
            )
        """

    query = (
        "SELECT DISTINCT subject "
        "FROM chapter_materials "
        "WHERE " + base_condition + member_clause +
        " ORDER BY subject"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [row["subject"] for row in rows]


def fetch_class_division_filters(
    admin_id: int,
    *,
    class_filter: Optional[str] = None,
    member_id: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Return distinct standards (classes) and divisions for the admin.

    When a class_filter is provided, divisions are restricted to that class.
    """

    class_conditions = ["admin_id = %(admin_id)s", "TRIM(COALESCE(std, '')) <> ''"]
    params: Dict[str, Any] = {"admin_id": admin_id}

    division_conditions = ["admin_id = %(admin_id)s", "TRIM(COALESCE(division, '')) <> ''"]

    if member_id is not None:
        params["member_id"] = member_id
        class_conditions.append("assigned_member_id = %(member_id)s")
        division_conditions.append("assigned_member_id = %(member_id)s")

    if class_filter:
        params["class_filter"] = class_filter
        division_conditions.append(
            "(TRIM(std) = %(class_filter)s OR LOWER(TRIM(std)) = LOWER(TRIM(%(class_filter)s)))"
        )

    class_query = (
        "SELECT std FROM ("
        "    SELECT DISTINCT TRIM(std) AS std "
        "    FROM student_roster_entries "
        "    WHERE " + " AND ".join(class_conditions) +
        ") AS distinct_classes "
        "ORDER BY "
        "    CASE WHEN std ~ '^[0-9]+$' THEN LPAD(std, 5, '0') ELSE UPPER(std) END, "
        "    std"
    )

    division_query = (
        "SELECT division FROM ("
        "    SELECT DISTINCT TRIM(division) AS division "
        "    FROM student_roster_entries "
        "    WHERE " + " AND ".join(division_conditions) +
        ") AS distinct_divisions "
        "ORDER BY UPPER(division)"
    )

    with get_pg_cursor() as cur:
        cur.execute(class_query, params)
        classes = [row["std"] for row in cur.fetchall() if row["std"]]

        cur.execute(division_query, params)
        divisions = [row["division"] for row in cur.fetchall() if row["division"]]

    return {"classes": classes, "divisions": divisions}