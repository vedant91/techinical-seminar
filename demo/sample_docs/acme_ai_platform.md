# Acme AI Platform

Product specification for the Acme AI Platform, version 3.2. This specification
describes the architecture, security controls, pricing model, and roadmap of the
platform. Acme ships quarterly updates and publishes this document to customers.

## 1. Overview

The Acme AI Platform is a managed service for building and operating AI
assistants over private document collections. Customers upload documents, the
platform indexes them, and end users ask natural language questions. Every
answer is traceable to the underlying source documents.

The platform serves two audiences: administrators who configure document
libraries, and end users who query them. Both interact through a single web
console and a REST API.

## 2. Architecture

The platform is split into a control plane and a data plane, with all customer
content isolated by workspace.

### 2.1 Control Plane

The control plane handles authentication, workspace configuration, and billing.
It stores no customer documents. All control plane traffic runs inside a
private network.

### 2.2 Data Plane

The data plane performs document indexing and query processing. Indexing jobs
run on isolated worker nodes and are horizontally scaled by document volume.
Query latency is p95 under 800 milliseconds for a ten-thousand document library.

Key components of the data plane include:

- An ingestion gateway that normalizes PDF, DOCX, HTML, and Markdown documents.
- A chunking service that preserves headings, tables, and list structure.
- A hybrid retrieval engine that combines dense vector search with a structure index.
- A validation service that scores retrieved context before answer generation.

## 3. Security & Encryption

All customer data is encrypted at rest and in transit. Data at rest is
protected with AES-256-GCM, and data in transit uses TLS 1.3. Encryption keys
are managed in a dedicated key management service, and keys are rotated every
ninety days automatically.

| Layer | Standard |
|---|---|
| Data at rest | AES-256-GCM |
| Data in transit | TLS 1.3 |
| Key management | Dedicated KMS, 90-day rotation |
| Hashing | SHA-256 for integrity checks |

Encryption standards are reviewed annually against industry guidance. The
platform also supports customer-managed keys for enterprise workspaces, which
allow customers to rotate keys on demand without platform intervention.

## 4. Pricing

Pricing is tiered by monthly active users. The Starter tier costs $99 per
month and includes ten seats and one document library. The Growth tier costs
$499 per month with fifty seats. The Enterprise tier costs $1,999 per month
with unlimited seats and dedicated support.

All tiers include the hybrid retrieval engine and the validation pipeline.
Annual billing receives a fifteen percent discount. Pricing is reviewed each
quarter, and customers are notified sixty days before any change.

## 5. Roadmap

Planned capabilities for the next two quarters:

- Multi-document cross-referencing in answers.
- Fine-grained access control per document section.
- Batch evaluation tooling for retrieval quality.
- Regional data residency in three additional regions.

The roadmap is indicative and may change based on customer feedback.
