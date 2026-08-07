# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in `medlatents`, please report it
privately by emailing **liamchalcroft@gmail.com** with details and steps to
reproduce. Please do not open a public issue for security-sensitive reports.

You can expect an acknowledgement within a few business days. Once the issue is
confirmed and a fix is prepared, we will coordinate disclosure.

## Note on Model Checkpoints

`medlatents` is a research library for generative modeling of medical imagery.
It does not perform authentication or process untrusted network input. However,
loading model checkpoints with `torch.load` deserializes pickled data, which can
execute arbitrary code, so **only load checkpoints from sources you trust.**
