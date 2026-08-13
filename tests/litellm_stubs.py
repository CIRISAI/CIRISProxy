"""
Stand-in litellm.proxy modules, installed before any hook is imported.

`hooks/custom_auth.py` imports `litellm.proxy._types` and
`litellm.proxy.proxy_server` at module scope. Those pull in the full proxy
server — a heavy import that needs a database and config the test run does not
have — so the suite substitutes lightweight stubs.

The substitution has to happen before the first `import hooks.custom_auth`
anywhere in the run, which is why this module is imported from `conftest.py`:
pytest loads conftest before collecting any test module, so the ordering holds
no matter how individual test files sort their imports. Doing it inline in a
test file instead forces module-level imports below executable code, which is
both a lint violation and a trap for the next person who runs an import sorter
over it.

Importing this module is what installs the stubs; it is idempotent.
"""

import sys
from unittest.mock import MagicMock


class MockUserAPIKeyAuth:
    """Stand-in for litellm.proxy._types.UserAPIKeyAuth."""

    def __init__(self, api_key: str, user_id: str):
        self.api_key = api_key
        self.user_id = user_id


class MockProxyException(Exception):
    """Stand-in for litellm.proxy.proxy_server.ProxyException."""

    def __init__(self, message: str, type: str, param: str = None, code: int = 401):
        self.message = message
        self.type = type
        self.param = param
        self.code = code
        super().__init__(message)


class MockRequest:
    """Minimal FastAPI Request stand-in with a headers mapping."""

    def __init__(self):
        self.headers = {}


def install() -> None:
    """Install the stub modules into sys.modules. Safe to call repeatedly."""
    mock_litellm_types = MagicMock()
    mock_litellm_types.UserAPIKeyAuth = MockUserAPIKeyAuth

    mock_litellm_proxy = MagicMock()
    mock_litellm_proxy.ProxyException = MockProxyException

    sys.modules["litellm.proxy._types"] = mock_litellm_types
    sys.modules["litellm.proxy.proxy_server"] = mock_litellm_proxy


install()
