"""
Unit tests for CIRISBillingCallback.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from hooks.billing_callback import CIRISBillingCallback


class TestParseUserKey:
    """Tests for _parse_user_key method."""

    def test_parse_google_prefixed_key(self):
        """Test parsing 'google:{user_id}' format."""
        callback = CIRISBillingCallback()
        provider, user_id = callback._parse_user_key("google:user-123")

        assert provider == "oauth:google"
        assert user_id == "user-123"

    def test_parse_other_provider_key(self):
        """Test parsing '{provider}:{user_id}' format."""
        callback = CIRISBillingCallback()
        provider, user_id = callback._parse_user_key("github:user-456")

        assert provider == "oauth:github"
        assert user_id == "user-456"

    def test_parse_plain_user_id(self):
        """Test parsing plain user ID (assumes Google)."""
        callback = CIRISBillingCallback()
        provider, user_id = callback._parse_user_key("user-789")

        assert provider == "oauth:google"
        assert user_id == "user-789"

    def test_parse_empty_key(self):
        """Test parsing empty API key."""
        callback = CIRISBillingCallback()
        provider, user_id = callback._parse_user_key("")

        assert provider == "oauth:google"
        assert user_id == ""

    def test_parse_key_with_multiple_colons(self):
        """Test parsing key with multiple colons."""
        callback = CIRISBillingCallback()
        provider, user_id = callback._parse_user_key("google:user:with:colons")

        assert provider == "oauth:google"
        assert user_id == "user:with:colons"


class TestPreCallHook:
    """Tests for async_pre_call_hook method."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @respx.mock
    @pytest.mark.asyncio
    async def test_pre_call_authorizes_new_interaction(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test that first call for interaction checks auth."""
        # Mock the credits/check endpoint (new API)
        auth_route = respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(
                200,
                json={
                    "has_credit": True,
                    "free_uses_remaining": 0,
                    "credits_remaining": 10,
                    "daily_free_uses_remaining": 0,
                },
            )
        )

        await callback.async_pre_call_hook(
            user_api_key_dict=sample_user_api_key_dict,
            cache=None,
            data=sample_request_data,
            call_type="completion",
        )

        # Verify auth was called
        assert auth_route.called
        assert auth_route.call_count == 1

        # Verify metadata was set
        assert sample_request_data["metadata"]["_ciris_external_id"] == "test-user-123"
        assert sample_request_data["metadata"]["_ciris_interaction_id"] == "int-test-456"
        assert sample_request_data["metadata"]["_ciris_oauth_provider"] == "oauth:google"

        # Verify interaction is cached
        assert "int-test-456" in callback._authorized_interactions

    @respx.mock
    @pytest.mark.asyncio
    async def test_pre_call_skips_auth_for_cached_interaction(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test that subsequent calls skip auth check."""
        # Pre-cache the interaction
        callback._authorized_interactions["int-test-456"] = True

        # Mock the auth endpoint (should not be called)
        auth_route = respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(200, json={"has_credit": True})
        )

        await callback.async_pre_call_hook(
            user_api_key_dict=sample_user_api_key_dict,
            cache=None,
            data=sample_request_data,
            call_type="completion",
        )

        # Verify auth was NOT called
        assert not auth_route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_pre_call_rejects_unauthorized_user(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test that unauthorized users are rejected."""
        # Mock the auth endpoint to deny
        respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(
                200,
                json={
                    "has_credit": False,
                    "free_uses_remaining": 0,
                    "credits_remaining": 0,
                    "daily_free_uses_remaining": 0,
                },
            )
        )

        with pytest.raises(Exception) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=sample_user_api_key_dict,
                cache=None,
                data=sample_request_data,
                call_type="completion",
            )

        assert "Insufficient credits" in str(exc_info.value)
        assert "int-test-456" not in callback._authorized_interactions

    @pytest.mark.asyncio
    async def test_pre_call_rejects_missing_interaction_id(
        self, callback, sample_user_api_key_dict
    ):
        """Test that missing interaction_id is rejected."""
        data = {
            "model": "groq/llama-3.1-70b",
            "messages": [],
            "metadata": {},  # No interaction_id
        }

        with pytest.raises(Exception) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=sample_user_api_key_dict,
                cache=None,
                data=data,
                call_type="completion",
            )

        assert "interaction_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pre_call_rejects_invalid_api_key(self, callback):
        """Test that missing user ID in API key is rejected."""
        data = {
            "model": "groq/llama-3.1-70b",
            "messages": [],
            "metadata": {"interaction_id": "int-123"},
        }

        with pytest.raises(Exception) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict={"api_key": ""},  # Empty key
                cache=None,
                data=data,
                call_type="completion",
            )

        assert "Invalid API key" in str(exc_info.value)

    @respx.mock
    @pytest.mark.asyncio
    async def test_pre_call_handles_billing_service_error(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test handling of billing service errors."""
        # Mock the auth endpoint to return 500
        respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(Exception) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=sample_user_api_key_dict,
                cache=None,
                data=sample_request_data,
                call_type="completion",
            )

        assert "Billing service error" in str(exc_info.value)

    @respx.mock
    @pytest.mark.asyncio
    async def test_pre_call_handles_network_error(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test handling of network errors."""
        # Mock network failure
        respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with pytest.raises(Exception) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=sample_user_api_key_dict,
                cache=None,
                data=sample_request_data,
                call_type="completion",
            )

        assert "unavailable" in str(exc_info.value).lower()


class TestLogSuccessEvent:
    """Tests for async_log_success_event method."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @respx.mock
    @pytest.mark.asyncio
    async def test_success_event_charges_interaction(
        self, callback, sample_response_obj, mock_litellm_params, billing_url
    ):
        """Test that success event calls charge endpoint."""
        # Mock the charge endpoint (new API)
        charge_route = respx.post(f"{billing_url}/v1/billing/charges").mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "charge-123",
                    "amount_minor": 1,
                    "balance_after": 9,
                },
            )
        )
        # Also mock the usage streaming endpoint
        respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        kwargs = {"litellm_params": mock_litellm_params, "model": "groq/llama-3.1-70b"}
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=2)

        await callback.async_log_success_event(
            kwargs=kwargs,
            response_obj=sample_response_obj,
            start_time=start_time,
            end_time=end_time,
        )

        # Verify charge was called
        assert charge_route.called
        assert charge_route.call_count == 1

        # Verify request body
        request_json = charge_route.calls[0].request.content.decode()
        assert "int-test-456" in request_json  # interaction_id
        assert "test-user-123" in request_json  # external_id

    @respx.mock
    @pytest.mark.asyncio
    async def test_success_event_aggregates_usage(
        self, callback, sample_response_obj, mock_litellm_params, billing_url
    ):
        """Test that success event aggregates usage data."""
        # Mock the charge endpoint
        respx.post(f"{billing_url}/v1/billing/charges").mock(
            return_value=httpx.Response(201, json={"id": "charge-123"})
        )
        # Mock the usage streaming endpoint
        respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        # Initialize interaction tracking
        callback._interaction_usage["int-test-456"]["external_id"] = "test-user-123"

        kwargs = {"litellm_params": mock_litellm_params, "model": "groq/llama-3.1-70b"}
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=2)

        await callback.async_log_success_event(
            kwargs=kwargs,
            response_obj=sample_response_obj,
            start_time=start_time,
            end_time=end_time,
        )

        # Verify usage was aggregated
        usage_data = callback._interaction_usage["int-test-456"]
        assert usage_data["llm_calls"] == 1
        assert usage_data["prompt_tokens"] == 100
        assert usage_data["completion_tokens"] == 50
        assert "groq/llama-3.1-70b" in usage_data["models"]
        assert usage_data["cost_cents"] == pytest.approx(0.12, rel=0.01)
        assert usage_data["duration_ms"] >= 2000

    @respx.mock
    @pytest.mark.asyncio
    async def test_success_event_handles_multiple_calls(
        self, callback, sample_response_obj, mock_litellm_params, billing_url
    ):
        """Test that multiple calls aggregate correctly."""
        # Mock the charge endpoint
        respx.post(f"{billing_url}/v1/billing/charges").mock(
            return_value=httpx.Response(201, json={"id": "charge-123"})
        )
        # Mock the usage streaming endpoint
        respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        callback._interaction_usage["int-test-456"]["external_id"] = "test-user-123"

        kwargs = {"litellm_params": mock_litellm_params, "model": "groq/llama-3.1-70b"}
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=1)

        # Simulate 3 LLM calls
        for _ in range(3):
            await callback.async_log_success_event(
                kwargs=kwargs,
                response_obj=sample_response_obj,
                start_time=start_time,
                end_time=end_time,
            )

        # Verify aggregation
        usage_data = callback._interaction_usage["int-test-456"]
        assert usage_data["llm_calls"] == 3
        assert usage_data["prompt_tokens"] == 300  # 100 * 3
        assert usage_data["completion_tokens"] == 150  # 50 * 3

    @pytest.mark.asyncio
    async def test_success_event_skips_without_metadata(self, callback, sample_response_obj):
        """Test that missing metadata is handled gracefully."""
        kwargs = {"litellm_params": {"metadata": {}}}  # No CIRIS metadata

        # Should not raise
        await callback.async_log_success_event(
            kwargs=kwargs,
            response_obj=sample_response_obj,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )

        # No usage should be tracked
        assert len(callback._interaction_usage) == 0


class TestLogFailureEvent:
    """Tests for async_log_failure_event method."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @pytest.mark.asyncio
    async def test_failure_event_increments_error_count(
        self, callback, mock_litellm_params
    ):
        """Test that failure events increment error count."""
        callback._interaction_usage["int-test-456"]["external_id"] = "test-user-123"

        kwargs = {"litellm_params": mock_litellm_params}

        await callback.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )

        assert callback._interaction_usage["int-test-456"]["errors"] == 1

        # Call again
        await callback.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )

        assert callback._interaction_usage["int-test-456"]["errors"] == 2


class TestFinalizeInteraction:
    """Tests for finalize_interaction method."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @respx.mock
    @pytest.mark.asyncio
    async def test_finalize_logs_usage(self, callback, billing_url):
        """Test that finalize_interaction logs usage to billing."""
        # Set up usage data
        callback._interaction_usage["int-test-456"] = {
            "external_id": "test-user-123",
            "llm_calls": 45,
            "prompt_tokens": 80000,
            "completion_tokens": 5000,
            "models": {"groq/llama-3.1-70b", "together/mixtral"},
            "cost_cents": 12.5,
            "duration_ms": 8500,
            "errors": 2,
            "fallbacks": 1,
            "start_time": datetime.now(timezone.utc) - timedelta(seconds=10),
        }
        callback._authorized_interactions["int-test-456"] = True

        # Mock usage endpoint
        usage_route = respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        await callback.finalize_interaction(
            external_id="test-user-123",
            interaction_id="int-test-456",
        )

        # Verify usage was logged
        assert usage_route.called
        request_json = usage_route.calls[0].request.content.decode()
        assert "45" in request_json  # total_llm_calls
        assert "80000" in request_json  # prompt_tokens
        assert "5000" in request_json  # completion_tokens

        # Verify cleanup
        assert "int-test-456" not in callback._interaction_usage
        assert "int-test-456" not in callback._authorized_interactions

    @pytest.mark.asyncio
    async def test_finalize_handles_missing_interaction(self, callback):
        """Test that finalizing unknown interaction is handled."""
        # Should not raise
        await callback.finalize_interaction(
            external_id="test-user",
            interaction_id="unknown-interaction",
        )


class TestCleanupStaleInteractions:
    """Tests for cleanup_stale_interactions method."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @respx.mock
    @pytest.mark.asyncio
    async def test_cleanup_removes_stale_interactions(self, callback, billing_url):
        """Test that stale interactions are cleaned up."""
        # Set up a stale interaction (2 hours old)
        callback._interaction_usage["stale-int-123"] = {
            "external_id": "test-user",
            "llm_calls": 10,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "models": {"groq/llama-3.1-70b"},
            "cost_cents": 1.0,
            "duration_ms": 5000,
            "errors": 0,
            "fallbacks": 0,
            "start_time": datetime.now(timezone.utc) - timedelta(hours=2),
        }
        callback._authorized_interactions["stale-int-123"] = True

        # Set up a fresh interaction
        callback._interaction_usage["fresh-int-456"] = {
            "external_id": "test-user",
            "llm_calls": 5,
            "prompt_tokens": 500,
            "completion_tokens": 250,
            "models": set(),
            "cost_cents": 0.5,
            "duration_ms": 2000,
            "errors": 0,
            "fallbacks": 0,
            "start_time": datetime.now(timezone.utc),
        }

        # Mock usage endpoint
        respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        # Clean up interactions older than 1 hour
        await callback.cleanup_stale_interactions(max_age_seconds=3600)

        # Stale should be removed
        assert "stale-int-123" not in callback._interaction_usage
        assert "stale-int-123" not in callback._authorized_interactions

        # Fresh should remain
        assert "fresh-int-456" in callback._interaction_usage


class TestInteractionLimits:
    """Tests for interaction limit enforcement (stale/excessive calls)."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @respx.mock
    @pytest.mark.asyncio
    async def test_stale_interaction_triggers_new_charge(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test that interaction reused after 5 minutes gets a continuation ID."""
        from hooks.billing_callback import MAX_INTERACTION_AGE_SECONDS

        # Set up an old interaction (6 minutes ago)
        interaction_id = sample_request_data["metadata"]["interaction_id"]
        callback._interaction_usage[interaction_id] = {
            "external_id": "test-user-123",
            "llm_calls": 10,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "models": set(),
            "cost_cents": 1.0,
            "duration_ms": 5000,
            "errors": 0,
            "fallbacks": 0,
            "start_time": datetime.now(timezone.utc) - timedelta(seconds=MAX_INTERACTION_AGE_SECONDS + 60),
        }
        callback._authorized_interactions[interaction_id] = True

        # Mock auth endpoint for the new continuation ID
        auth_route = respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(
                200, json={"has_credit": True, "credits_remaining": 10, "free_uses_remaining": 0, "daily_free_uses_remaining": 0}
            )
        )

        # Mock usage endpoint for finalization of stale interaction
        usage_route = respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        await callback.async_pre_call_hook(
            user_api_key_dict=sample_user_api_key_dict,
            cache=None,
            data=sample_request_data,
            call_type="completion",
        )

        # Should have called auth (for new continuation ID)
        assert auth_route.called

        # Should have logged usage for the stale interaction
        assert usage_route.called

        # The stored interaction_id should be a continuation ID
        stored_id = sample_request_data["metadata"]["_ciris_interaction_id"]
        assert "__cont_stale_" in stored_id
        assert stored_id.startswith(interaction_id)

    @respx.mock
    @pytest.mark.asyncio
    async def test_excessive_calls_triggers_new_charge(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test that exceeding 80 LLM calls triggers a continuation charge."""
        from hooks.billing_callback import MAX_LLM_CALLS_PER_INTERACTION

        # Set up an interaction with 80 calls already
        interaction_id = sample_request_data["metadata"]["interaction_id"]
        callback._interaction_usage[interaction_id] = {
            "external_id": "test-user-123",
            "llm_calls": MAX_LLM_CALLS_PER_INTERACTION,  # At the limit
            "prompt_tokens": 100000,
            "completion_tokens": 50000,
            "models": set(),
            "cost_cents": 10.0,
            "duration_ms": 60000,
            "errors": 0,
            "fallbacks": 0,
            "start_time": datetime.now(timezone.utc) - timedelta(seconds=60),  # Recent
        }
        callback._authorized_interactions[interaction_id] = True

        # Mock auth endpoint for the new continuation ID
        auth_route = respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(
                200, json={"has_credit": True, "credits_remaining": 9, "free_uses_remaining": 0, "daily_free_uses_remaining": 0}
            )
        )

        # Mock usage endpoint for finalization of excessive-calls interaction
        usage_route = respx.post(f"{billing_url}/v1/billing/litellm/usage").mock(
            return_value=httpx.Response(200, json={"logged": True})
        )

        await callback.async_pre_call_hook(
            user_api_key_dict=sample_user_api_key_dict,
            cache=None,
            data=sample_request_data,
            call_type="completion",
        )

        # Should have called auth (for new continuation ID)
        assert auth_route.called

        # Should have logged usage for the exceeded interaction
        assert usage_route.called

        # The stored interaction_id should be a continuation ID
        stored_id = sample_request_data["metadata"]["_ciris_interaction_id"]
        assert "__cont_calls" in stored_id
        assert stored_id.startswith(interaction_id)

    @respx.mock
    @pytest.mark.asyncio
    async def test_normal_interaction_not_affected(
        self, callback, sample_user_api_key_dict, sample_request_data, billing_url
    ):
        """Test that normal interactions within limits are not affected."""
        # Set up a normal interaction (2 minutes old, 10 calls)
        interaction_id = sample_request_data["metadata"]["interaction_id"]
        callback._interaction_usage[interaction_id] = {
            "external_id": "test-user-123",
            "llm_calls": 10,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "models": set(),
            "cost_cents": 1.0,
            "duration_ms": 5000,
            "errors": 0,
            "fallbacks": 0,
            "start_time": datetime.now(timezone.utc) - timedelta(seconds=120),  # 2 minutes
        }
        callback._authorized_interactions[interaction_id] = True

        # Auth should NOT be called (cached)
        auth_route = respx.post(f"{billing_url}/v1/billing/credits/check").mock(
            return_value=httpx.Response(
                200, json={"has_credit": True, "credits_remaining": 10, "free_uses_remaining": 0, "daily_free_uses_remaining": 0}
            )
        )

        await callback.async_pre_call_hook(
            user_api_key_dict=sample_user_api_key_dict,
            cache=None,
            data=sample_request_data,
            call_type="completion",
        )

        # Should NOT have called auth (using cache)
        assert not auth_route.called

        # The interaction_id should be unchanged
        stored_id = sample_request_data["metadata"]["_ciris_interaction_id"]
        assert stored_id == interaction_id
        assert "__cont_" not in stored_id


class TestClientManagement:
    """Tests for HTTP client management."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return CIRISBillingCallback()

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, callback):
        """Test that _get_client creates a client."""
        client = await callback._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed

        # Cleanup
        await callback.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, callback):
        """Test that _get_client reuses existing client."""
        client1 = await callback._get_client()
        client2 = await callback._get_client()

        assert client1 is client2

        # Cleanup
        await callback.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self, callback):
        """Test that close() closes the client."""
        client = await callback._get_client()
        assert not client.is_closed

        await callback.close()

        assert callback._client is None
