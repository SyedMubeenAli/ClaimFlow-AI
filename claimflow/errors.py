"""
Error types used across ClaimFlow.

Every error that can reach a person carries a message written for that person.
The UI shows `user_message`; the full technical detail goes to the log.
Unexpected exceptions (plain Python errors) are caught at the edges and shown
as a generic message, so a stack trace never reaches the screen.
"""

from __future__ import annotations


class ClaimFlowError(Exception):
    """Base class. `message` says what went wrong, `hint` says how to fix it."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        return f"{self.message} {self.hint}" if self.hint else self.message


class ConfigError(ClaimFlowError):
    """A setting is missing or has an invalid value."""


class InvalidInputError(ClaimFlowError):
    """The person filled in a form incorrectly."""


class DatabaseError(ClaimFlowError):
    """SQLite failed, or a record was not found."""


class UploadError(ClaimFlowError):
    """An uploaded file was rejected."""


class IndexingError(ClaimFlowError):
    """A policy could not be turned into a searchable index."""


class WorkflowError(ClaimFlowError):
    """An action is not allowed in the claim's current state."""


class AgentError(ClaimFlowError):
    """An agent broke its contract or could not finish."""


class LLMError(ClaimFlowError):
    """Base class for problems talking to a language model."""


class LLMAuthError(LLMError):
    """The provider rejected the API key."""


class LLMRateLimitError(LLMError):
    """The provider says we are sending too many requests."""


class LLMUnavailableError(LLMError):
    """Network failure, timeout or a 5xx from the provider."""


class LLMResponseError(LLMError):
    """The provider answered, but the answer is unusable."""


class LLMCacheMiss(LLMError):
    """Demo mode needed a recorded answer and none exists."""


# Only these are worth retrying. Auth and bad-response errors will fail again.
RETRYABLE_LLM_ERRORS = (LLMRateLimitError, LLMUnavailableError)
