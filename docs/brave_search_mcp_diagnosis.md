# Brave Search MCP Diagnosis

Date: 2026-07-16

## Symptom

The configured Brave Search MCP tool returns:

```text
fetch failed
```

This happens even for a minimal query such as:

```text
test
```

## MCP Configuration Check

`codex mcp list` and `codex mcp get brave-search` show:

```text
Name: brave-search
Command: npx
Args: -y @brave/brave-search-mcp-server --transport stdio
Env: BRAVE_API_KEY=*****
Status: enabled
```

The API key is present in `~/.codex/config.toml`. The key itself was not printed
or copied into project files.

## Findings

### 1. Initial npm cache issue

Running the package initially failed with:

```text
npm EPERM ... C:\Users\Yjia\AppData\Local\npm-cache\_cacache\tmp\...
```

After running the `npx` command with approval, the package could start and show
its help output. This fixed the npm-cache/package-start issue.

### 2. Brave API key is valid

A direct Python HTTPS request to Brave Search API with the configured key
returned HTTP `200` when run with approved network access.

Therefore:

- the key is not empty;
- the key format is acceptable;
- the Brave API endpoint can return valid search JSON.

### 3. Node fetch still fails

Node's built-in `fetch` fails against the same Brave API endpoint:

```text
TypeError: fetch failed
ConnectTimeoutError: Connect Timeout Error
```

The attempted addresses included unexpected IP ranges such as:

```text
31.13.70.9
2a03:2880:...
162.125.32.10
```

These are not expected for `api.search.brave.com`.

### 4. DNS / network behavior differs by runtime

Observed behavior:

- `Resolve-DnsName api.search.brave.com` returned unexpected addresses.
- Node `dns.lookup()` returned unexpected addresses.
- Python `socket.getaddrinfo()` returned a different IPv4 address in one run.
- Python direct HTTP worked only with approved network access.
- `curl.exe` timed out from the restricted shell path.

Current interpretation:

```text
The Brave MCP failure is caused by the Node-based MCP server's network/DNS path
in this environment, not by a missing MCP registration or missing API key.
```

## Current Workaround

Added:

```text
scripts/brave_search_direct.py
```

Usage:

```powershell
python scripts\brave_search_direct.py "low light raw denoising noise synthesis" --count 5
```

This script:

- reads the same `BRAVE_API_KEY` from environment or `~/.codex/config.toml`;
- calls Brave Search API directly with Python `urllib`;
- prints title, URL, and description;
- can output raw JSON with `--raw`.

Because normal shell network access is restricted, this fallback may still need
approved execution when used for live searches.

Validated on 2026-07-16 with:

```powershell
python scripts\brave_search_direct.py "Learning to See in the Dark low light raw denoising SID dataset" --count 2
```

The fallback returned Brave results including the SID project page and a recent
SIED arXiv result. A same-turn retry of the Brave MCP `brave_web_search` tool
still returned `fetch failed`.

## Long-Term Fix Options

1. Run Codex/MCP with a network policy where Node `fetch` can access
   `https://api.search.brave.com`.
2. Replace the Node Brave MCP server with a local Python MCP wrapper.
3. Configure a reliable DNS/proxy route for Node processes.
4. Continue using arXiv MCP plus `scripts/brave_search_direct.py` for web search
   until the Node network path is fixed.

## Practical Decision

For the ICCD project, use:

- arXiv MCP for paper discovery;
- official webpages or normal web browsing for source verification;
- `scripts/brave_search_direct.py` as a Brave API fallback;
- avoid treating failed Brave MCP results as evidence of missing literature.
