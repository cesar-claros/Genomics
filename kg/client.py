"""
Neo4j driver wrapper.

Reads connection info from environment variables (NEO4J_URI, NEO4J_USER,
NEO4J_PASSWORD, optional NEO4J_DATABASE), supports context-manager use,
and runs the schema constraints on first connection so MERGE upserts
behave correctly.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from neo4j import Driver, GraphDatabase, Session

from .schema import CONSTRAINTS_CYPHER


class Neo4jClient:
    """Thin wrapper around neo4j.GraphDatabase.driver with lifecycle helpers."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
    ):
        self.uri = uri
        self.user = user
        self.database = database
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._constraints_applied = False

    @classmethod
    def from_env(cls) -> "Neo4jClient":
        """Construct from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE."""
        uri = os.environ.get("NEO4J_URI")
        if not uri:
            raise ValueError(
                "NEO4J_URI is not set. Configure NEO4J_URI, NEO4J_USER, and "
                "NEO4J_PASSWORD in code/.env or the environment."
            )
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD is not set.")
        return cls(
            uri=uri,
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=password,
            database=os.environ.get("NEO4J_DATABASE"),
        )

    def ensure_constraints(self) -> None:
        """Apply uniqueness constraints. Idempotent. Called automatically by
        session() on first use; safe to call explicitly."""
        if self._constraints_applied:
            return
        with self._driver.session(database=self.database) as session:
            for stmt in CONSTRAINTS_CYPHER:
                session.run(stmt)
        self._constraints_applied = True

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a Neo4j Session, applying constraints on first use."""
        self.ensure_constraints()
        with self._driver.session(database=self.database) as session:
            yield session

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
