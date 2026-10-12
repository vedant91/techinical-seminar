# Acme Security Audit

Independent security audit of the Acme AI Platform, conducted in July 2026 by
an external audit firm. This report documents the audit scope, methodology,
findings, and recommendations.

## 1. Scope

The audit covered the platform's production environment, its document
processing pipeline, and the encryption controls described in the product
specification. Code review focused on the ingestion gateway and the retrieval
engine.

## 2. Methodology

The auditors reviewed architecture documentation, interviewed platform
engineers, and performed automated vulnerability scans. Configuration was
compared against the encryption standards published in the product
specification.

## 3. Key Findings

The audit confirmed that most security claims in the product specification are
implemented as described. Encryption at rest uses AES-256-GCM as promised, and
TLS 1.3 is enforced on all external endpoints.

### 3.1 Findings Summary

| Finding | Severity | Status |
|---|---|---|
| Encryption at rest matches specification | Informational | Confirmed |
| TLS 1.3 enforced on all endpoints | Informational | Confirmed |
| Key rotation interval is 90 days as stated | Informational | Confirmed |
| Two API endpoints lacked rate limiting | Medium | Open |
| Log retention shorter than documented | Low | Open |

### 3.2 Medium Severity Finding

Two administrative API endpoints lack rate limiting. An attacker with valid
credentials could brute-force parameter values at high volume. The audit team
recommends adding per-token throttling before the next release.

## 4. Recommendations

The audit team recommends the following actions:

- Add rate limiting to the two administrative endpoints identified above.
- Extend audit log retention to match the documented ninety-day window.
- Re-run the automated vulnerability scan after each platform release.
- Publish a security changelog alongside each product specification update.

Overall, the audit found no evidence that customer data was exposed or that
encryption controls were bypassed.
