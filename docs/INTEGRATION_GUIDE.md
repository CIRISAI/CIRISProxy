# CIRISProxy Integration Guide

Integration guide for the CIRIS Agent team to connect mobile clients to the LLM proxy.

## Endpoint

```
https://llm.ciris.ai
```

## Authentication

All requests must include an `Authorization` header with a **Google ID Token**:

```
Authorization: Bearer {google_id_token}
```

The token is a JWT (~1200 characters) obtained from Google Sign-In on the client.

### How It Works

1. Android app calls `GoogleSignIn.getClient().silentSignIn()`
2. Gets `GoogleSignInAccount.idToken` (a JWT)
3. Sends as `Authorization: Bearer {idToken}`
4. Proxy verifies JWT signature against Google's public keys
5. Extracts Google user ID (`sub` claim) for billing

### Token Properties

- **Cryptographically signed** by Google - cannot be forged
- **Expires in ~1 hour** - client must refresh periodically
- **Audience-locked** - only valid for CIRIS app's client ID

### Getting the Token (Android)

```kotlin
val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
    .requestIdToken("265882853697-l421ndojcs5nm7lkln53jj29kf7kck91.apps.googleusercontent.com")
    .requestEmail()
    .build()

val account = GoogleSignIn.getLastSignedInAccount(context)
val idToken = account?.idToken  // Use this in Authorization header
```

### Token Refresh

Google ID tokens expire in ~1 hour. Handle 401 responses by refreshing:

```kotlin
GoogleSignIn.getClient(context, gso).silentSignIn()
    .addOnSuccessListener { account ->
        val newToken = account.idToken
        // Retry request with new token
    }
```

## Required Headers

| Header | Value | Description |
|--------|-------|-------------|
| `Authorization` | `Bearer {google_id_token}` | Google ID Token (JWT) |
| `Content-Type` | `application/json` | Required for all POST requests |

## Required Metadata

Every request MUST include `interaction_id` in the metadata. This is critical for the credit system:

```json
{
  "metadata": {
    "interaction_id": "unique-interaction-uuid"
  }
}
```

**Important**:
- Generate ONE `interaction_id` per user interaction (message → response cycle)
- Reuse the SAME `interaction_id` for all LLM calls within that interaction
- The billing system charges 1 credit per unique `interaction_id`, regardless of how many LLM calls are made

## Available Models

### Recommended Models

| Model Name | Provider | Best For |
|------------|----------|----------|
| `default` | Groq | General use, fast responses |
| `fast` | Groq | Quick responses, lower quality |
| `groq/llama-3.3-70b` | Groq | High quality reasoning |
| `llama-4-maverick` | Together AI | Advanced reasoning (when available) |

### All Available Models

```
# Groq (fast, recommended)
groq/llama-3.3-70b
groq/llama-3.1-8b
groq/llama-4-maverick

# Together AI
together/llama-3.1-70b
together/llama-3.1-8b
together/qwen-2.5-72b
llama-4-maverick

# OpenAI (fallback)
openai/gpt-4o-mini
openai/gpt-4o

# Aliases
default  → groq/llama-3.3-70b-versatile
fast     → groq/llama-3.1-8b-instant
```

## API Endpoints

### Chat Completions

```
POST https://llm.ciris.ai/v1/chat/completions
```

**Request:**
```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "metadata": {
    "interaction_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1732596000,
  "model": "groq/llama-3.3-70b-versatile",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

### Streaming

Add `"stream": true` to enable streaming responses:

```json
{
  "model": "default",
  "messages": [...],
  "stream": true,
  "metadata": {
    "interaction_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### List Models

```
GET https://llm.ciris.ai/v1/models
Authorization: Bearer google:{user_id}
```

### Health Check

```
GET https://llm.ciris.ai/health/liveliness
```

No authentication required. Returns `"I'm alive!"` if healthy.

## Code Examples

### Python (httpx)

```python
import httpx
import uuid

class CIRISProxyClient:
    def __init__(self, google_user_id: str):
        self.base_url = "https://llm.ciris.ai"
        self.headers = {
            "Authorization": f"Bearer google:{google_user_id}",
            "Content-Type": "application/json"
        }

    async def chat(
        self,
        messages: list[dict],
        interaction_id: str,
        model: str = "default",
        stream: bool = False
    ):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "metadata": {
                        "interaction_id": interaction_id
                    }
                },
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()

# Usage
async def main():
    client = CIRISProxyClient("118234567890123456789")

    # Generate ONE interaction_id for the entire user interaction
    interaction_id = str(uuid.uuid4())

    # First LLM call in the interaction
    response1 = await client.chat(
        messages=[{"role": "user", "content": "What's 2+2?"}],
        interaction_id=interaction_id
    )

    # Second LLM call (tool use, etc.) - SAME interaction_id
    response2 = await client.chat(
        messages=[
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": response1["choices"][0]["message"]["content"]},
            {"role": "user", "content": "Now multiply by 3"}
        ],
        interaction_id=interaction_id  # Same ID = still 1 credit
    )
```

### Kotlin (Android)

```kotlin
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

class CIRISProxyClient(private val googleUserId: String) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private val baseUrl = "https://llm.ciris.ai"
    private val jsonMediaType = "application/json".toMediaType()

    fun chat(
        messages: List<Message>,
        interactionId: String,
        model: String = "default"
    ): ChatResponse {
        val messagesArray = JSONArray().apply {
            messages.forEach { msg ->
                put(JSONObject().apply {
                    put("role", msg.role)
                    put("content", msg.content)
                })
            }
        }

        val requestBody = JSONObject().apply {
            put("model", model)
            put("messages", messagesArray)
            put("metadata", JSONObject().apply {
                put("interaction_id", interactionId)
            })
        }.toString().toRequestBody(jsonMediaType)

        val request = Request.Builder()
            .url("$baseUrl/v1/chat/completions")
            .addHeader("Authorization", "Bearer google:$googleUserId")
            .addHeader("Content-Type", "application/json")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw CIRISProxyException(response.code, response.body?.string())
            }
            return parseResponse(response.body?.string())
        }
    }

    companion object {
        fun generateInteractionId(): String = UUID.randomUUID().toString()
    }
}

// Usage
fun handleUserMessage(userMessage: String) {
    val client = CIRISProxyClient(googleSignInAccount.id)

    // One interaction_id for the entire user interaction
    val interactionId = CIRISProxyClient.generateInteractionId()

    // All LLM calls use the same interaction_id
    val response = client.chat(
        messages = listOf(Message("user", userMessage)),
        interactionId = interactionId
    )
}
```

### Swift (iOS)

```swift
import Foundation

class CIRISProxyClient {
    private let baseURL = "https://llm.ciris.ai"
    private let googleUserId: String

    init(googleUserId: String) {
        self.googleUserId = googleUserId
    }

    func chat(
        messages: [[String: String]],
        interactionId: String,
        model: String = "default"
    ) async throws -> ChatResponse {
        guard let url = URL(string: "\(baseURL)/v1/chat/completions") else {
            throw CIRISProxyError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("Bearer google:\(googleUserId)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120

        let body: [String: Any] = [
            "model": model,
            "messages": messages,
            "metadata": [
                "interaction_id": interactionId
            ]
        ]

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw CIRISProxyError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            throw CIRISProxyError.httpError(httpResponse.statusCode, String(data: data, encoding: .utf8))
        }

        return try JSONDecoder().decode(ChatResponse.self, from: data)
    }

    static func generateInteractionId() -> String {
        return UUID().uuidString
    }
}

// Usage
func handleUserMessage(_ message: String) async {
    let client = CIRISProxyClient(googleUserId: GIDSignIn.sharedInstance.currentUser?.userID ?? "")

    // One interaction_id per user interaction
    let interactionId = CIRISProxyClient.generateInteractionId()

    do {
        let response = try await client.chat(
            messages: [["role": "user", "content": message]],
            interactionId: interactionId
        )
        // Handle response
    } catch {
        // Handle error
    }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

### Common Errors

| HTTP Code | Error Type | Cause | Solution |
|-----------|------------|-------|----------|
| 401 | `auth_error` | Invalid or missing API key | Check Authorization header format |
| 401 | `auth_error` | User not found in billing | User needs to register in CIRIS app |
| 402 | `insufficient_credits` | No credits remaining | User needs to purchase credits |
| 400 | `invalid_request` | Missing interaction_id | Include metadata.interaction_id |
| 429 | `rate_limit` | Too many requests | Implement exponential backoff |
| 500 | `server_error` | Upstream provider error | Retry with backoff, try different model |
| 503 | `service_unavailable` | Billing service down | Retry later |

### Retry Strategy

```python
import asyncio
import random

async def chat_with_retry(client, messages, interaction_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.chat(messages, interaction_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry
                wait = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)
            elif e.response.status_code >= 500:
                # Server error - retry with backoff
                wait = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)
            else:
                # Client error - don't retry
                raise
    raise Exception("Max retries exceeded")
```

## Credit System

### How Credits Work

1. **1 credit = 1 interaction** (user message → agent response)
2. An interaction may include 12-70+ LLM calls (tool use, reasoning, etc.)
3. All LLM calls with the same `interaction_id` are billed as ONE credit
4. The billing system uses idempotency - first call charges, rest are no-ops

### Credit Flow

```
User sends message
    ↓
Generate interaction_id (UUID)
    ↓
First LLM call → Billing auth check → 1 credit charged
    ↓
Tool use LLM call → Same interaction_id → No additional charge
    ↓
Reasoning LLM call → Same interaction_id → No additional charge
    ↓
Final response LLM call → Same interaction_id → No additional charge
    ↓
User receives response
    ↓
Total charged: 1 credit
```

### Best Practices

1. **Generate interaction_id early** - Create it when user sends a message
2. **Reuse consistently** - Pass the same ID to ALL LLM calls in the interaction
3. **Don't reuse across interactions** - New user message = new interaction_id
4. **Store for debugging** - Log interaction_id for troubleshooting

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per second (per IP) | 10 |
| Burst allowance | 20 requests |
| Request timeout | 120 seconds |
| Max request body | 10 MB |

## Timeouts

Configure your HTTP client with appropriate timeouts:

| Timeout | Recommended |
|---------|-------------|
| Connect | 10 seconds |
| Read | 120 seconds |
| Write | 30 seconds |

LLM responses can take 10-60 seconds depending on prompt length and model.

## Fallback Chain

The proxy automatically handles model fallbacks:

```
default → together/llama-3.1-70b → openai/gpt-4o-mini
groq/llama-3.3-70b → together/llama-3.1-70b → openai/gpt-4o-mini
llama-4-maverick → groq/llama-4-maverick → together/llama-3.1-70b
```

You don't need to implement fallback logic - the proxy handles it transparently.

## Testing

### Health Check

```bash
curl https://llm.ciris.ai/health/liveliness
# Returns: "I'm alive!"
```

### Test Request (requires valid user)

```bash
curl -X POST https://llm.ciris.ai/v1/chat/completions \
  -H "Authorization: Bearer google:YOUR_GOOGLE_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"interaction_id": "test-123"}
  }'
```

## Support

- **Proxy Issues**: Check https://llm.ciris.ai/health/liveliness
- **Billing Issues**: Contact CIRISBilling team
- **Integration Help**: See CLAUDE.md in CIRISProxy repo

## Changelog

| Date | Change |
|------|--------|
| 2025-11-26 | Initial TLS deployment at llm.ciris.ai |
| 2025-11-26 | Default model changed to groq/llama-3.3-70b |
