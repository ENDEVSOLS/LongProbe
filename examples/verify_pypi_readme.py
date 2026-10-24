"""Verify demo links and the versioned public descriptions in built distributions."""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


def verify(root: Path, *, check_public: bool = False) -> None:
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = config["project"]["version"]
    image_base = config["tool"]["hatch"]["metadata"]["hooks"]["custom"]["image-base-url"]
    image_base = image_base.format(version=version).rstrip("/")
    readme = (root / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]\n]*\]\(([^\s)]+\.gif(?:\?[^\s)]*)?)\)", readme)
    if not links:
        raise ValueError("The checkout README must contain demo GIF links.")
    public_root = image_base.rsplit("/", 1)[0] + "/"
    expected = readme
    paths = []
    for link in links:
        if link.startswith("assets/"):
            path = urlsplit(link).path
        elif link.startswith(public_root):
            _, path = link.removeprefix(public_root).split("/", 1)
            path = urlsplit(path).path
        else:
            raise ValueError(
                f"Demo GIF must use local assets or the configured public repository: {link}"
            )
        if not path.startswith("assets/"):
            raise ValueError(f"Unexpected demo asset path: {path}")
        paths.append(path)
        expected = expected.replace(f"]({link})", f"]({image_base}/{path})")

    def check_metadata(raw: bytes, label: str) -> None:
        metadata = BytesParser().parsebytes(raw)
        if metadata["Version"] != version:
            raise ValueError(f"{label}: wrong distribution version")
        if metadata["Description-Content-Type"] != "text/markdown":
            raise ValueError(f"{label}: description must be Markdown")
        if metadata.get_payload(decode=True).decode("utf-8").strip() != expected.strip():
            raise ValueError(f"{label}: description does not match the versioned public README")
        print(f"{label}: {len(paths)} versioned public GIF links verified")

    wheel_path = root / "dist" / f"longprobe-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_path = f"longprobe-{version}.dist-info/METADATA"
        check_metadata(wheel.read(metadata_path), "Wheel")
    with tarfile.open(root / "dist" / f"longprobe-{version}.tar.gz") as archive:
        prefix = f"longprobe-{version}/"
        metadata_file = archive.extractfile(prefix + "PKG-INFO")
        readme_file = archive.extractfile(prefix + "README.md")
        if metadata_file is None or readme_file is None:
            raise ValueError("Source archive is missing its description or README")
        check_metadata(metadata_file.read(), "Source archive")
        if readme_file.read().decode("utf-8") != readme:
            raise ValueError("The source archive must preserve the checkout README")
        archive.getmember(prefix + "hatch_build.py")
    print("Checkout README preserved; the source archive includes the rebuild hook.")
    if check_public:
        for path in paths:
            local = (root / path).read_bytes()
            url = f"{image_base}/{path}"
            try:
                with urlopen(url, timeout=15) as response:
                    public = response.read(len(local) + 1)
            except OSError as error:
                raise ValueError(
                    f"Public GIF unavailable: {url}. Push the matching tag and assets before publishing."
                ) from error
            if hashlib.sha256(public).digest() != hashlib.sha256(local).digest():
                raise ValueError(f"Public GIF does not match the local release asset: {url}")
        print(f"All {len(paths)} public release GIFs match the local assets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-public",
        action="store_true",
        help="Check tagged public GIFs against local release assets",
    )
    arguments = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], check_public=arguments.check_public)
