"""
Helpers for locating mutable runtime configuration files.

These helpers make PyInstaller deployments easier by preferring config files
next to the executable while still working in a source checkout.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def get_runtime_root() -> Path:
    """Return the directory where mutable runtime files should live."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_runtime_path(path_str: str) -> Path:
    """Resolve absolute paths directly and relative paths against runtime root."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return get_runtime_root() / candidate


def _find_existing_path(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def ensure_runtime_file(
    default_name: str,
    template_name: Optional[str] = None,
    env_var_name: Optional[str] = None,
    fallback_names: Iterable[str] = (),
) -> Path:
    """
    Resolve the preferred runtime file path.

    Resolution order:
    1. Explicit env var path, if provided.
    2. Preferred runtime file under the runtime root.
    3. If the preferred file is missing and a template exists, copy template.
    4. Fallback file names under runtime root / cwd.
    5. Return the preferred runtime file path even if it does not exist yet.
    """
    if env_var_name:
        explicit_value = os.getenv(env_var_name)
        if explicit_value:
            explicit_path = resolve_runtime_path(explicit_value)
            logger.info("Using config path from %s: %s", env_var_name, explicit_path)
            return explicit_path

    runtime_root = get_runtime_root()
    preferred_path = runtime_root / default_name
    if preferred_path.exists():
        return preferred_path

    if template_name:
        template_candidates = [
            runtime_root / template_name,
            Path.cwd() / template_name,
        ]
        template_path = _find_existing_path(template_candidates)
        if template_path and template_path != preferred_path:
            preferred_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template_path, preferred_path)
            logger.info(
                "Created runtime config from template: %s -> %s",
                template_path,
                preferred_path,
            )
            return preferred_path

    fallback_candidates = []
    for fallback_name in fallback_names:
        fallback_candidates.append(runtime_root / fallback_name)
        fallback_candidates.append(Path.cwd() / fallback_name)

    existing_fallback = _find_existing_path(fallback_candidates)
    if existing_fallback:
        logger.info("Falling back to existing config file: %s", existing_fallback)
        return existing_fallback

    return preferred_path
