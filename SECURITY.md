# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

## Reporting a Vulnerability

The LongProbe team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**security@endevsols.com**

### What to Include

To help us better understand and resolve the issue, please include as much of the following information as possible:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths of source file(s)** related to the manifestation of the issue
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Updates**: We will send you regular updates about our progress
- **Timeline**: We aim to resolve critical issues within 7 days
- **Credit**: If you wish, we will publicly credit you for the discovery once the issue is resolved

### Security Best Practices for Users

When using LongProbe in production:

1. **Environment Variables**: Store sensitive credentials (API keys, database passwords) in environment variables, not in configuration files
   ```yaml
   # Good
   api_key: "${OPENAI_API_KEY}"
   
   # Bad
   api_key: "sk-proj-abc123..."
   ```

2. **File Permissions**: Restrict access to configuration files containing sensitive data
   ```bash
   chmod 600 longprobe.yaml
   chmod 700 .longprobe/
   ```

3. **Network Security**: When using HTTP adapters, always use HTTPS in production
   ```yaml
   retriever:
     type: "http"
     url: "https://api.example.com/retrieve"  # Use HTTPS
   ```

4. **Input Validation**: Validate and sanitize user inputs before using them in queries

5. **Dependency Updates**: Keep LongProbe and its dependencies up to date
   ```bash
   pip install --upgrade longprobe
   ```

6. **Baseline Database**: Protect your baseline database file from unauthorized access
   ```bash
   chmod 600 .longprobe/baselines.db
   ```

7. **CI/CD Secrets**: Use GitHub Secrets or equivalent for API keys in CI/CD pipelines
   ```yaml
   env:
     OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
   ```

## Known Security Considerations

### 1. LLM API Keys

LongProbe may use LLM APIs (OpenAI, Anthropic, etc.) for question generation. Always:
- Store API keys in environment variables
- Never commit API keys to version control
- Rotate keys regularly
- Use least-privilege API keys when possible

### 2. Vector Store Credentials

When connecting to vector databases:
- Use read-only credentials when possible
- Implement network-level access controls
- Use encrypted connections (TLS/SSL)

### 3. HTTP Adapter

The HTTP adapter makes requests to external endpoints:
- Validate SSL certificates (don't disable verification)
- Use authentication headers securely
- Be cautious with untrusted endpoints
- Implement rate limiting on your API endpoints

### 4. Document Parsing

When using the document parser with untrusted files:
- Be aware that parsing PDFs/DOCX can execute embedded code
- Run parsing in isolated environments for untrusted documents
- Validate file types before parsing

### 5. Baseline Database and CI Retention

The SQLite baseline database stores test results and evaluation contracts:
- Contains full retrieval results, query contracts, and chunk content for reproducibility and strict diff comparisons
- May include sensitive proprietary information from your indexed documents
- Protect with appropriate local file permissions (`chmod 600 .longprobe/baselines.db`)
- Maintain in trusted storage; do not upload SQLite baselines to unauthenticated public CI artifact stores
- Shared CI outputs (`longprobe-comment.md`, annotations, and default `longprobe-result.json`) redact question text and chunk evidence by default

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

We will coordinate the disclosure with you and credit you in the release notes (unless you prefer to remain anonymous).

## Security Updates

Security updates will be released as patch versions (e.g., 0.2.1) and announced via:

- GitHub Security Advisories
- Release notes in CHANGELOG.md
- PyPI release notes

Subscribe to [GitHub releases](https://github.com/ENDEVSOLS/LongProbe/releases) to stay informed.

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or email security@endevsols.com.

---

**Last Updated**: November 2026
