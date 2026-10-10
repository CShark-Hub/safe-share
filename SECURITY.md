# Security policy

Safe Share redacts a live screen before it is shared, so a flaw can expose exactly what
it is meant to hide. Please report security issues privately.

## Reporting

Use GitHub private vulnerability reporting: **Security → Report a vulnerability** on this
repository. Do not open a public issue.

Please don't include real personal data or credentials in a report; use synthetic values.

## In scope

- Unmasked or partly masked sensitive content reaching the Safe Share window: a raw
  frame, a mask applied to the wrong frame, or output after Pause, Stop or an error.
- Screenshots, recognized text or detected names written to logs or disk.
- Any network access other than the one-time model download, which runs before capture.

Missed detections (a name or term the detector does not recognize) are known limitations;
report them as regular issues using synthetic examples.

## Support

This is a hackathon project with no supported versions and no response-time commitment.
Reports are handled on a best-effort basis on `main`.
