from __future__ import annotations

import base64
import getpass

import bcrypt


def main() -> None:
    password = getpass.getpass("Password to hash: ")
    if not password:
        raise SystemExit("Password cannot be empty.")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    hash_text = hashed.decode("utf-8")
    b64_text = base64.b64encode(hash_text.encode("utf-8")).decode("utf-8")
    print(f"Raw bcrypt hash: {hash_text}")
    print(f"Compose-safe APP_PASSWORD_HASH value: base64:{b64_text}")


if __name__ == "__main__":
    main()
