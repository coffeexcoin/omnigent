# Runner MCP ownership

The runner owns MCP connectors and stdio subprocesses. The API server's
`POST /v1/sessions/{id}/mcp` endpoint evaluates policy, pins the authenticated
turn actor, and delegates `tools/list` / `tools/call` to the runner's
`/mcp/execute` endpoint.

## Actor-aware credentials

Deployments can inject an `McpCredentialResolver` into `RunnerMcpManager` (or
through `runner._entry.create_app(mcp_credential_resolver=...)`). The resolver
runs for every schema lookup and tool call and returns `McpCredential` with:

- transport material (`headers` for HTTP or `env` for stdio),
- a non-secret credential `generation`, and
- either `scope="actor"` or an explicit `scope="service"` identity.

Actor-scoped HTTP connectors are pooled by server config, the complete canonical
turn actor, and credential generation. Actor-scoped stdio connectors additionally
include the session in their pool partition; actor takeover or credential rotation
retires the previous connector process for that session without replacing the
runner sandbox. Service-scoped connectors are pooled by server config, explicit
service identity, and generation. A resolver must bump the generation whenever
credential material changes. Credential material is merged only into the effective
connector config; pool hashes, status snapshots, and logs contain only non-secret
identity/generation partitions.

Prewarming is disabled when a credential resolver is configured because no turn
actor exists at runner startup. The first real request resolves and starts the
correct connector lazily.
