# Security Policy

## Supported versions

iai-mcp is pre-1.0 and experimental. Only the latest tagged release on `main` receives fixes. Older tags are not maintained.

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes (latest) |
| 0.3.x   | Security fixes only |
| < 0.3   | No |

## Reporting an issue

Please do **not** open a public issue for defects that have security implications. This includes anything that could:

- Disclose stored memories to a third party.
- Allow recovery of data without the configured passphrase.
- Cause the daemon to execute arbitrary code from untrusted input.
- Bypass the local-only network posture.

### How to report

Use GitHub Security Advisories on the repository:

https://github.com/CodeAbra/iai-mcp/security/advisories/new

Include:

- A description of the issue and its impact.
- Steps to reproduce, or a proof-of-concept if available.
- Affected version (`iai-mcp --version`).
- Your environment (macOS version, Python version).

You will receive an acknowledgement within a reasonable window. There is no formal SLA; this is a single-maintainer project. Reports are handled on a best-effort basis.

## Disclosure

Once a fix is available, the advisory is published with credit to the reporter (unless anonymity is requested). Backports to older releases are not guaranteed.

## Threat model and scope

iai-mcp runs locally and is designed around the following assumptions:

- The host machine is trusted. An attacker with local code execution as the user can read the encryption key (`~/.iai-mcp/.key`) and the unlocked store.
- The MCP host (Claude Code, Claude Desktop, etc.) is trusted. Captured turns include whatever content the host sends.
- No network exposure. The daemon listens on a UNIX socket only. Any change that adds a TCP listener, HTTP server, or remote sync is out of scope and should be discussed in a public issue first.

### In scope for security reports

- Disclosure of stored records without the passphrase.
- Weakening of the AES-256-GCM encryption-at-rest (key derivation, nonce reuse, etc.).
- Arbitrary code execution triggered by captured content.
- Path traversal or unsafe file operations under `~/.iai-mcp/`.
- Injection into the MCP wrapper that affects the host process.

### Out of scope

- Social engineering of the user.
- Physical access to an unlocked machine.
- Defects requiring the user to manually set insecure file permissions on `~/.iai-mcp/.key`.
- Denial-of-service against a single user's local daemon by their own process (the daemon is single-user by design).

## Cryptographic notes

- Records are encrypted at rest with AES-256-GCM via the `cryptography` library (pyca/cryptography).
- The key is derived from a user-provided passphrase using a standard KDF and stored at `~/.iai-mcp/.key` with mode 0600.
- The OS keychain is used opportunistically via the `keyring` package on platforms where it is available.

Cryptographic primitives are intentionally not hand-rolled. If you spot a misuse of an existing primitive, please report via the advisory flow above.
