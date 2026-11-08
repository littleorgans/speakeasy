"""The road test Names list as a backend receives it (src/names.ts owns and canonicalizes it).

The host sends the list already canonical, so this check changes nothing it
accepts: a list of at most MAX_NAMES distinct, trimmed, non blank strings of at
most MAX_NAME_BYTES UTF8 bytes, with no control characters or commas.
"""

import hashlib
import json
import re

MAX_NAMES = 64
MAX_NAME_BYTES = 128
# C0, DEL and C1 controls, and the comma that joins a prompt.
REFUSED = re.compile("[\x00-\x1f\x7f-\x9f,]")


def names_option(value):
    """The validated names, as a new list; raises ValueError naming the first bad name."""
    if not isinstance(value, list) or len(value) > MAX_NAMES:
        raise ValueError(f"names must be a list of at most {MAX_NAMES} strings")
    seen = set()
    for name in value:
        if not isinstance(name, str):
            raise ValueError("names must be strings")
        if not name or name != name.strip() or REFUSED.search(name):
            raise ValueError(f"name {name!r} must be trimmed, non blank, without control characters or commas")
        if len(name.encode("utf-8")) > MAX_NAME_BYTES:
            raise ValueError(f"name {name!r} exceeds {MAX_NAME_BYTES} UTF8 bytes")
        if name in seen:
            raise ValueError(f"name {name!r} is repeated")
        seen.add(name)
    return list(value)


def names_digest(names):
    """sha256 of the compact JSON array, the bytes JSON.stringify and serde_json give."""
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
