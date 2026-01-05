# SSH Configuration

This directory contains SSH public keys for accessing the AuroraAI system.

## Authorized Keys

The `authorized_keys` file contains public keys that are authorized to access this system.

### Current Keys

1. **kenderovemil@msi-vector17hxaia2xwjg**
   - Type: ED25519 256-bit
   - Fingerprint: `SHA256:GV8GS0EPtXj/Z9TCKKGChWtArChjkXC/Yg/nNMl4aTw`

## Usage

To add a new SSH key:
1. Generate the public key fingerprint to verify it matches
2. Add the full public key to the `authorized_keys` file
3. Ensure the file has proper permissions (typically 600 or 644)

## Security

- Only add trusted public keys
- Regularly review and remove unused keys
- Keep private keys secure and never commit them to the repository
