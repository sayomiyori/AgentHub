# Embedding request retry boundary

When an embedding provider is configured, construct the existing Gemini SDK
client with one total attempt and a 20,000 millisecond per-request timeout.
A failed or uncertain request must not be automatically retried by SDK defaults.
Use the existing fixed EmbeddingUnavailable error; preserve offline fallback and
tenant-provider guards. No provider/model/key change or new config/dependency.
The timeout is an SDK transport bound, not an absolute document/job deadline;
large document processing and task cancellation remain separately unverified.
One SDK attempt does not guarantee one HTTP exchange if the SDK follows a
redirect; redirect policy is unchanged in this increment.

Verify constructor options and use the real SDK with local HTTP transport for
503, 429 and timeout paths: one attempt, static sanitized error, no fallback.
Never contact a real provider. Keep prior parser/worker/tenant tests green.
