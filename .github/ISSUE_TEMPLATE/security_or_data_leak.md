---
name: Security or data leak report
about: Report security concerns without exposing private information publicly
title: "[Security/Data Leak]: "
labels: security
assignees: ""
---

## Before Posting

Do not paste secrets, tokens, credentials, private keys, raw private data,
private source code, raw diffs, model weights, or generated datasets into a
public issue. Redact sensitive values and share only the minimum safe summary.

If the report includes an active secret, private dataset, private source file,
or exploitable vulnerability, use GitHub private vulnerability reporting if it
is enabled for this repository, or contact the maintainers out-of-band before
sharing details publicly.

## Safe Summary

Describe the issue without including secrets or raw private data.

## Affected Area

- [ ] Generated artifacts or model artifacts
- [ ] Raw data intake
- [ ] Local run history
- [ ] CLI output or logs
- [ ] Documentation or examples
- [ ] Other:

## Redaction Checklist

- [ ] I redacted tokens, credentials, private keys, and account identifiers.
- [ ] I did not attach raw private data, model weights, generated datasets, or
      private source code.
- [ ] I replaced sensitive paths or values with safe placeholders.

