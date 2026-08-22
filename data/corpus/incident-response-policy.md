# Meridian Pay — Incident Response Policy

## Severity Levels

Meridian Pay classifies incidents into three severity levels:

- **SEV1**: Complete outage or confirmed customer data breach. Affects all customers.
- **SEV2**: Partial outage or degraded service affecting a subset of customers.
- **SEV3**: Minor issue with a workaround; no meaningful customer impact.

## Response Times

For a SEV1 incident, an engineer must acknowledge the alert within 15 minutes. For SEV2,
the acknowledgment window is 30 minutes. SEV3 incidents must be acknowledged within one
business day.

## Incident Commander

Every SEV1 and SEV2 incident has a single Incident Commander who coordinates the
response. The Incident Commander is the on-call engineering lead by default.

## Customer Communication

For any SEV1 incident, the company posts a public status update within 60 minutes and
updates it at least once every hour until resolution.

## Postmortems

Every SEV1 and SEV2 incident requires a written blameless postmortem, completed within
five business days of resolution. SEV3 incidents do not require a postmortem.

## On-Call Rotation

The engineering on-call rotation is one week long and rotates every Monday at 9:00 AM
Central Time.
