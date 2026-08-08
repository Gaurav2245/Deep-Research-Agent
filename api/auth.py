"""Optional API-key authentication.

If the API_KEY environment variable is unset, no authentication is enforced
(the default for local development). If it is set, every request to a
protected router must send it in the X-API-Key header.
"""
from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str | None = Security(_api_key_header)) -> None:
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return  # Auth disabled: no API_KEY configured.
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
