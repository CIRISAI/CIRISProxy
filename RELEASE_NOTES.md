# CIRISProxy Release Notes

## v0.2.0 - 2025-12-16

**Image:** `ghcr.io/cirisai/cirisproxy:latest`
**Commits:** `2690554`, `928ea62`

---

### For Agent Team

**No changes required** - all updates are backward compatible.

#### New Features

1. **Exa AI Search (ZDR-Compliant)**
   - Primary search provider now uses Exa AI with Zero Data Retention
   - Brave Search remains as automatic fallback
   - Response format unchanged - `results.web.results[]` structure preserved
   - New optional request params: `category`, `search_type`, `include_domains`, `exclude_domains`

2. **Enhanced Error Diagnostics**
   - LLM errors now include provider identification
   - Helps debug which backend returns malformed responses
   - No client-side changes needed

#### Response Format (unchanged)
```json
{
  "results": {
    "web": {
      "results": [
        {"title": "...", "url": "...", "description": "..."}
      ]
    },
    "provider": "exa"  // NEW: indicates which provider was used
  }
}
```

---

### For Lens Team

#### New Log Fields in `llm_error` Events

Error logs now include provider debugging info:

| Field | Type | Description |
|-------|------|-------------|
| `provider` | string | `groq`, `together`, `openrouter`, `openai`, or `unknown` |
| `actual_model` | string | Full model path (e.g., `openrouter/meta-llama/llama-4-maverick`) |
| `api_base` | string | Provider endpoint URL (truncated to 50 chars) |

#### Sample Query: Find Failing Provider

```sql
SELECT provider, COUNT(*) as error_count,
       MAX(error) as last_error,
       MAX(timestamp) as last_seen
FROM cirislens.service_logs
WHERE service_name = 'cirisproxy'
  AND event = 'llm_error'
  AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY provider
ORDER BY error_count DESC;
```

#### Sample Query: Recent Errors with Provider

```sql
SELECT timestamp, provider, actual_model, error, interaction_id
FROM cirislens.service_logs
WHERE service_name = 'cirisproxy'
  AND event = 'llm_error'
ORDER BY timestamp DESC
LIMIT 50;
```

#### Dashboard Suggestion

Add a panel grouping errors by `provider` to quickly identify which LLM backend is returning malformed JSON for complex schemas.

---

### For Bridge Team (Deployment)

**Environment variable changes:**

| Variable | Required | Description |
|----------|----------|-------------|
| `EXA_API_KEY` | Yes* | Exa AI API key (get from https://dashboard.exa.ai) |
| `SEARCH_PROVIDER` | No | `auto` (default), `exa`, or `brave` |

*Required if web search functionality is needed. Falls back to `BRAVE_API_KEY` if not set.

**Deploy command:**
```bash
cd ~/CIRISBridge/ansible
ansible-playbook -i inventory/production.yml playbooks/site.yml --tags proxy
```

---

### Known Issue Under Investigation

ActionSelectionPDMA occasionally receives malformed JSON like:
```json
{'type': 'type: ', 'type: ': 'type: '}
```

The new provider logging will help identify which backend (groq/together/openrouter) is causing this. Once identified, we can exclude it from the fallback chain for complex structured output prompts.
