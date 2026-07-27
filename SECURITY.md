# Security Policy

## Reporting A Vulnerability

Please report suspected security issues privately by emailing:

```text
security@methodintelligence.io
```

Do not open a public GitHub issue for vulnerabilities, worker-token leaks, pairing-code handling problems, authorization bypasses, artifact-validation bypasses, or anything that could expose private Midom project data.

When reporting, include:

- A short description of the issue.
- The affected plugin version or commit.
- Steps to reproduce, if safe to share.
- The Midom deployment type involved, such as local development or production.
- Relevant logs with secrets removed.

Do not include live pairing codes, worker tokens, Midom session cookies, private project files, or private generated media unless specifically requested through a secure channel.

## Supported Versions

Security fixes are expected to target the current public version of the Midom WanGP Bridge Plugin. Older development snapshots may not receive separate fixes.

## Security Model

The Midom WanGP Bridge Plugin is an untrusted local worker client from Midom's point of view. Users can modify the plugin source, so Midom server-side authorization and validation must remain authoritative.

The plugin should only receive scoped worker credentials. It should never require or store a user's normal Midom login credentials.

Worker tokens are intended to be scoped to a single worker, user, organization, and project. A token should only authorize worker routes needed for pairing, capability reporting, heartbeats, job polling, job claiming, claimed-job input downloads, progress updates, artifact uploads, job completion, failure reporting, and disconnect/revocation handling.

## Operator Guidance

- Keep `worker_config.json` private.
- Do not post WanGP console logs publicly without reviewing them for tokens, pairing codes, URLs, project identifiers, and local file paths.
- Pair only with Midom deployments and projects you trust.
- Use HTTPS for normal Midom deployments.
- Enable localhost or private LAN HTTP only for intentional development use.
- Revoke stale or suspicious workers from Midom.

## WanGP And Model Security

This plugin runs inside a local WanGP installation and calls WanGP internals. WanGP, model files, LoRAs, FFmpeg, Python packages, GPU drivers, and other local dependencies have their own security and licensing considerations. This plugin's MIT license does not change the license or security policy of WanGP or third-party model assets.
