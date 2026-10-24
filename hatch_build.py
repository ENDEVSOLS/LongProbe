"""Prepare the PyPI description without changing the checkout's README."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    """Pin local or public demo images to the versioned public asset repository."""

    def update(self, metadata: dict[str, Any]) -> None:
        readme = Path(self.root, "README.md").read_text(encoding="utf-8")
        image_base = self.config["image-base-url"].format(version=metadata["version"]).rstrip("/")
        public_root = image_base.rsplit("/", 1)[0]
        images = re.compile(
            r"(!\[[^\]\n]*\]\()(?:"
            + re.escape(public_root)
            + r"/[^/\s)]+/)?(assets/[^\s)]+\.gif)(?:\?[^\s)]*)?(\))"
        )
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": images.sub(
                lambda match: f"{match[1]}{image_base}/{match[2]}{match[3]}", readme
            ),
        }
