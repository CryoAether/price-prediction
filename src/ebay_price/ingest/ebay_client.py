from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EbayClient:
    """
    Placeholder interface. Replace list_completed and list_active with real eBay API calls.
    """

    def __init__(self, site: str = "EBAY_US"):
        self.site = site

    def list_completed(
        self, query: str, category_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """
        Return a list of raw listing dicts that ended (sold or not).
        Replace with eBay 'completed items' endpoint.
        """
        return []

    def list_active(
        self, query: str, category_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """
        Return a list of raw listing dicts that are currently active.
        """
        return []

    @staticmethod
    def load_local_jsonl(path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
