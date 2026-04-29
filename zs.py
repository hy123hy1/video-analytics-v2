#!/usr/bin/env python3
import sys
import time
import jwt

# Open PEM
private_key = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEICyt4em6w4gpPrZYOtRUCSq2iIoVgQTKeZHJVW2CSIUL
-----END PRIVATE KEY-----
"""

payload = {
    'iat': int(time.time()) - 30,
    'exp': int(time.time()) + 900,
    'sub': '2M2BEU76BV'
}
headers = {
    'kid': 'C8GYPMNG7B'
}

# Generate JWT
encoded_jwt = jwt.encode(payload, private_key, algorithm='EdDSA', headers = headers)

print(f"JWT:  {encoded_jwt}")
