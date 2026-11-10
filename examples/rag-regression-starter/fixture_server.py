"""Simulated 3-state local retrieval fixture for the LongProbe starter kit.

Operates in 3 states to demonstrate the pass -> fail -> fix workflow:
  - good: Known-good configuration (100% recall, 20/20 questions pass)
  - degraded: Simulated retrieval regression (drops 3 chunks, recall drops to 85%)
  - corrected: Corrected configuration (100% recall restored)

Runs locally without external services, database installations, or API keys.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

KNOWLEDGE_BASE: list[dict[str, Any]] = [
    # 1. Billing & Refund
    {
        "id": "q01_refund_window",
        "question": "What is the standard refund window for annual subscriptions?",
        "keywords": ["refund window", "standard refund"],
        "good_chunk": {
            "chunk_id": "refund-window",
            "content": "Annual subscriptions are eligible for a full refund within 30 days of initial purchase. After this 30-day window, subscriptions are non-refundable for the elapsed period.",
            "similarity": 1.0,
        },
        "degraded_chunk": {
            "chunk_id": "refund-generic",
            "content": "Our service plans can be upgraded or downgraded anytime through the billing settings portal.",
            "similarity": 0.35,
        },
        "degraded": True,
    },
    {
        "id": "q02_prorated_refund",
        "question": "Are prorated refunds offered after the initial 30-day period?",
        "keywords": ["prorated"],
        "good_chunk": {
            "chunk_id": "prorated-refund",
            "content": "Prorated refunds are available after 30 days upon written request to billing support, calculated on remaining whole unused months.",
            "similarity": 1.0,
        },
        "degraded_chunk": {
            "chunk_id": "billing-faq",
            "content": "Billing invoices can be downloaded directly from your customer dashboard in PDF format.",
            "similarity": 0.30,
        },
        "degraded": True,
    },
    {
        "id": "q03_payment_terms",
        "question": "What are the standard payment terms for enterprise invoices?",
        "keywords": ["payment terms", "enterprise invoices"],
        "good_chunk": {
            "chunk_id": "payment-terms",
            "content": "Standard enterprise invoice payment terms are net 30 days from date of issuance.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q04_early_discount",
        "question": "Is there a discount for paying enterprise invoices early?",
        "keywords": ["early payment", "discount"],
        "good_chunk": {
            "chunk_id": "early-discount",
            "content": "Enterprise clients receive a 2% early payment discount for settling invoices within 10 days.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q05_late_fee",
        "question": "What late fee applies to overdue invoices?",
        "keywords": ["late fee", "overdue"],
        "good_chunk": {
            "chunk_id": "late-fee",
            "content": "A 1.5% monthly late fee applies to any past due balance exceeding 30 calendar days.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    # 2. Security & Compliance
    {
        "id": "q06_encryption_rest",
        "question": "How is customer data encrypted at rest?",
        "keywords": ["encrypted at rest", "encryption at rest"],
        "good_chunk": {
            "chunk_id": "sec-chunk-01",
            "content": "Customer data at rest is encrypted using AES-256 standards with automated key rotation.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q07_encryption_transit",
        "question": "What protocol secures data in transit?",
        "keywords": ["data in transit", "transit"],
        "good_chunk": {
            "chunk_id": "sec-chunk-02",
            "content": "All network traffic in transit is strictly enforced with TLS 1.3 encryption.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q08_key_rotation",
        "question": "How often are encryption keys rotated?",
        "keywords": ["keys rotated", "key rotation"],
        "good_chunk": {
            "chunk_id": "sec-chunk-03",
            "content": "Master encryption keys are rotated automatically every 90 days in compliance with NIST standards.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q09_soc2_audit",
        "question": "How frequently does the platform undergo SOC 2 Type II audits?",
        "keywords": ["soc 2", "soc 2 type ii"],
        "good_chunk": {
            "chunk_id": "sec-chunk-04",
            "content": "Independent third-party SOC 2 Type II compliance audits are conducted on an annual basis.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q10_breach_notice",
        "question": "Within what timeframe are customers notified of a confirmed breach?",
        "keywords": ["confirmed breach", "breach"],
        "good_chunk": {
            "chunk_id": "sec-chunk-05",
            "content": "In the event of a confirmed security incident, affected customers are notified within 72 hours.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    # 3. Enterprise SLA
    {
        "id": "q11_uptime_sla",
        "question": "What is the guaranteed monthly uptime percentage for enterprise tiers?",
        "keywords": ["uptime percentage", "monthly uptime", "guaranteed monthly"],
        "good_chunk": {
            "chunk_id": "sla-uptime",
            "content": "Enterprise tier subscriptions include a guaranteed 99.95% monthly uptime service level agreement.",
            "similarity": 1.0,
        },
        "degraded_chunk": {
            "chunk_id": "sla-infrastructure",
            "content": "Our high-availability infrastructure is deployed across multiple geographic availability zones.",
            "similarity": 0.40,
        },
        "degraded": True,
    },
    {
        "id": "q12_sev1_response",
        "question": "What is the initial response time for a Severity 1 outage?",
        "keywords": ["severity 1", "sev1", "sev 1"],
        "good_chunk": {
            "chunk_id": "sla-sev1",
            "content": "Severity 1 critical incidents receive an initial engineering response within 15 minutes 24/7.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q13_sla_credit",
        "question": "What service credit is issued if monthly uptime falls below 99.9%?",
        "keywords": ["service credit", "sla credit"],
        "good_chunk": {
            "chunk_id": "sla-credit",
            "content": "Customers receive a 10% monthly service credit if monthly uptime drops below 99.9%.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q14_maintenance_notice",
        "question": "How much advance notice is given for scheduled maintenance?",
        "keywords": ["scheduled maintenance", "maintenance notice"],
        "good_chunk": {
            "chunk_id": "sla-maintenance",
            "content": "Scheduled system maintenance requires a minimum of 5 business days advance notification.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q15_rpo_target",
        "question": "What is the Recovery Point Objective (RPO) for production databases?",
        "keywords": ["recovery point objective", "rpo"],
        "good_chunk": {
            "chunk_id": "sla-rpo",
            "content": "The disaster recovery RPO for all primary production databases is guaranteed under 15 minutes.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    # 4. Account Governance & Legal
    {
        "id": "q16_authorized_signers",
        "question": "Which company officers are authorized to sign enterprise agreements?",
        "keywords": ["authorized to sign", "authorized signers", "officers"],
        "good_chunk": {
            "chunk_id": "gov-signers",
            "content": "Only authorized executive officers including the CEO, CFO, and General Counsel may execute agreements on behalf of the company.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q17_termination_notice",
        "question": "How many days of written notice are required to terminate a contract?",
        "keywords": ["terminate a contract", "termination notice", "terminate the master"],
        "good_chunk": {
            "chunk_id": "gov-termination",
            "content": "Either party may terminate the master services agreement by providing 30 days prior written notice.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q18_data_retention",
        "question": "How long are customer backups retained after account termination?",
        "keywords": ["backups retained", "data retention"],
        "good_chunk": {
            "chunk_id": "gov-retention",
            "content": "Customer data backups are retained for 60 days following account termination before permanent deletion.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q19_subprocessor_notice",
        "question": "How far in advance are customers notified before adding a new subprocessor?",
        "keywords": ["subprocessor"],
        "good_chunk": {
            "chunk_id": "gov-subprocessor",
            "content": "Customers receive at least 30 calendar days advance notice prior to onboarding new third-party subprocessors.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
    {
        "id": "q20_governing_law",
        "question": "Which jurisdiction's laws govern the master service agreement?",
        "keywords": ["jurisdiction", "governing law", "delaware"],
        "good_chunk": {
            "chunk_id": "gov-law",
            "content": "The master agreement and any disputes arising under it are governed by the laws of Delaware.",
            "similarity": 1.0,
        },
        "degraded": False,
    },
]


class RetrievalHandler(BaseHTTPRequestHandler):
    """Handles health checks and query retrieval requests."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server request logs for clean CLI output."""
        return

    def do_HEAD(self) -> None:
        """Health check probe."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self) -> None:
        """Health check endpoint and status inquiry."""
        mode = getattr(self.server, "mode", "good")
        body = json.dumps({"status": "ok", "mode": mode, "questions_loaded": len(KNOWLEDGE_BASE)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """Retrieve matching chunks for question, or toggle mode."""
        if self.path == "/mode":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
                new_mode = payload.get("mode", "good")
                if new_mode in {"good", "degraded", "corrected"}:
                    self.server.mode = new_mode  # type: ignore[attr-defined]
                    resp = json.dumps({"status": "ok", "mode": new_mode}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return
            except Exception:
                self.send_error(400, "Invalid mode payload")
                return

        # Main /retrieve path
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            query = str(payload.get("query", "")).strip().lower()
            top_k = max(1, int(payload.get("top_k", 5)))
        except (ValueError, KeyError, TypeError):
            self.send_error(400, "Expected JSON body with query and optional top_k")
            return

        mode = getattr(self.server, "mode", "good")
        chunks = self._match_query(query, mode)
        body = json.dumps({"data": {"chunks": chunks[:top_k]}}).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _match_query(self, query: str, mode: str) -> list[dict[str, Any]]:
        """Find matching chunks for the query according to the active mode."""
        query_norm = query.strip().lower()

        # 1. Check exact question match first
        for entry in KNOWLEDGE_BASE:
            if query_norm == entry["question"].strip().lower():
                if mode == "degraded" and entry.get("degraded", False):
                    return [entry["degraded_chunk"]]
                return [entry["good_chunk"]]

        # 2. Check keyword fallback
        for entry in KNOWLEDGE_BASE:
            keywords = entry.get("keywords", [])
            if any(kw.lower() in query_norm for kw in keywords):
                if mode == "degraded" and entry.get("degraded", False):
                    return [entry["degraded_chunk"]]
                return [entry["good_chunk"]]

        # Default fallback for unrecognised queries
        return [
            {
                "chunk_id": "fallback-chunk",
                "content": "General corporate overview: EnDevSols provides AI reliability engineering tooling.",
                "similarity": 0.2,
            }
        ]


def run_server(port: int = 8000, mode: str = "good") -> None:
    server_address = ("127.0.0.1", port)
    server = ThreadingHTTPServer(server_address, RetrievalHandler)
    server.mode = mode  # type: ignore[attr-defined]
    print(
        f"Simulated local retrieval fixture (mode={mode}) running at http://127.0.0.1:{port}/retrieve"
    )
    with suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local 3-state simulated retrieval fixture for LongProbe regression testing."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LONGPROBE_FIXTURE_PORT", 8000)),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--mode",
        choices=["good", "degraded", "corrected"],
        default=os.environ.get("LONGPROBE_FIXTURE_MODE", "good"),
        help="Fixture state: good (100%% recall), degraded (85%% recall), corrected (100%% recall)",
    )
    args = parser.parse_args()
    run_server(port=args.port, mode=args.mode)
