import importlib.metadata

# Read from installed package metadata rather than a second literal: this file
# and pyproject.toml have already drifted apart once, and db.writer_version()
# stamps the metadata value into every index it builds. One of the two had to
# stop being a source of truth.
try:
    __version__ = importlib.metadata.version("dejasearch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+source"
