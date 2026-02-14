"""
Account Rotator — Strategic multi-account credential rotation.

Distributes requests across multiple Google OAuth accounts to:
1. Maximize throughput (3 accounts = 3× quota)
2. Prevent abuse patterns (rate limiting per account)
3. Auto-recover from rate limits (429 backoff per account)

Design principles:
- Round-robin with per-account rate tracking
- Configurable max requests per minute per account
- Automatic cooldown on 429 responses
- Zero traffic to disabled/flagged accounts
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    """Tracks per-account usage and health."""
    email: str
    credential_path: str
    credentials: dict
    project_id: str
    
    # Rate tracking
    request_timestamps: list = field(default_factory=list)
    total_requests: int = 0
    total_errors: int = 0
    
    # Health
    is_healthy: bool = True
    cooldown_until: float = 0.0  # Unix timestamp
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        """Account is available if healthy and not in cooldown."""
        return self.is_healthy and time.time() > self.cooldown_until
    
    def requests_in_window(self, window_seconds: float = 60.0) -> int:
        """Count requests in the last N seconds."""
        cutoff = time.time() - window_seconds
        # Clean old timestamps while counting
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]
        return len(self.request_timestamps)
    
    def record_request(self):
        """Record a successful request."""
        now = time.time()
        self.request_timestamps.append(now)
        self.total_requests += 1
        self.consecutive_errors = 0
    
    def record_error(self, error_msg: str, is_rate_limit: bool = False):
        """Record an error and apply cooldown if needed."""
        self.total_errors += 1
        self.consecutive_errors += 1
        self.last_error = error_msg
        
        if is_rate_limit:
            # Exponential backoff: 30s, 60s, 120s, 300s max
            backoff = min(30 * (2 ** (self.consecutive_errors - 1)), 300)
            self.cooldown_until = time.time() + backoff
            logger.warning(
                f"Account {self.email} rate-limited. Cooldown {backoff}s "
                f"(consecutive errors: {self.consecutive_errors})"
            )
        elif self.consecutive_errors >= 5:
            # 5 consecutive non-rate-limit errors = disable account
            self.is_healthy = False
            logger.error(f"Account {self.email} disabled after {self.consecutive_errors} consecutive errors")


class AccountRotator:
    """
    Strategic multi-account rotation with rate limiting.
    
    Configuration via environment variables:
    - CREDENTIALS_DIR: Path to directory with credential JSON files (default: ./credentials)
    - MAX_RPM_PER_ACCOUNT: Max requests per minute per account (default: 10)
    - MIN_REQUEST_GAP_MS: Minimum milliseconds between requests to same account (default: 2000)
    """
    
    def __init__(
        self,
        credentials_dir: str = None,
        max_rpm_per_account: int = None,
        min_request_gap_ms: int = None,
    ):
        self.credentials_dir = credentials_dir or os.getenv("CREDENTIALS_DIR", "./credentials")
        self.max_rpm = max_rpm_per_account or int(os.getenv("MAX_RPM_PER_ACCOUNT", "10"))
        self.min_gap_s = (min_request_gap_ms or int(os.getenv("MIN_REQUEST_GAP_MS", "2000"))) / 1000.0
        
        self.accounts: list[AccountState] = []
        self._current_index = 0
        self._lock = threading.Lock()
        
        self._load_accounts()
    
    def _load_accounts(self):
        """Load all credential files from the credentials directory."""
        cred_path = Path(self.credentials_dir)
        if not cred_path.exists():
            logger.warning(f"Credentials directory not found: {self.credentials_dir}")
            return
        
        for cred_file in sorted(cred_path.glob("*.json")):
            try:
                with open(cred_file) as f:
                    creds = json.load(f)
                
                if not creds.get("refresh_token"):
                    logger.warning(f"Skipping {cred_file.name}: no refresh_token")
                    continue
                
                # Extract email from filename (name_at_domain_ext.json)
                email = cred_file.stem.replace("_at_", "@").replace("_", ".")
                
                account = AccountState(
                    email=email,
                    credential_path=str(cred_file),
                    credentials=creds,
                    project_id=creds.get("project_id", ""),
                )
                self.accounts.append(account)
                logger.info(f"Loaded account: {email} (project: {account.project_id})")
                
            except Exception as e:
                logger.error(f"Failed to load {cred_file}: {e}")
        
        logger.info(f"Account rotator initialized: {len(self.accounts)} accounts, {self.max_rpm} RPM/account")
    
    @property
    def available_accounts(self) -> list[AccountState]:
        """Get list of currently available accounts."""
        return [a for a in self.accounts if a.is_available]
    
    def get_next_account(self) -> Optional[AccountState]:
        """
        Get the next available account using round-robin with rate limiting.
        
        Returns None if all accounts are exhausted or in cooldown.
        """
        with self._lock:
            available = self.available_accounts
            if not available:
                logger.warning("No accounts available — all in cooldown or disabled")
                return None
            
            # Try each available account starting from current index
            for _ in range(len(available)):
                account = available[self._current_index % len(available)]
                self._current_index = (self._current_index + 1) % len(available)
                
                # Check per-minute rate limit
                if account.requests_in_window(60.0) >= self.max_rpm:
                    logger.debug(f"Account {account.email} at RPM limit ({self.max_rpm})")
                    continue
                
                # Check minimum gap between requests
                recent = account.requests_in_window(self.min_gap_s)
                if recent > 0:
                    logger.debug(f"Account {account.email} within min gap ({self.min_gap_s}s)")
                    continue
                
                return account
            
            logger.warning("All accounts at rate limit — consider reducing request volume")
            return None
    
    def record_success(self, account: AccountState):
        """Record a successful request for an account."""
        account.record_request()
    
    def record_error(self, account: AccountState, status_code: int, error_msg: str):
        """Record an error for an account."""
        is_rate_limit = status_code in (429, 503)
        account.record_error(error_msg, is_rate_limit=is_rate_limit)
    
    def get_stats(self) -> dict:
        """Get rotation statistics for monitoring."""
        return {
            "total_accounts": len(self.accounts),
            "available_accounts": len(self.available_accounts),
            "max_rpm_per_account": self.max_rpm,
            "min_request_gap_ms": int(self.min_gap_s * 1000),
            "accounts": [
                {
                    "email": a.email,
                    "is_available": a.is_available,
                    "is_healthy": a.is_healthy,
                    "total_requests": a.total_requests,
                    "total_errors": a.total_errors,
                    "rpm_current": a.requests_in_window(60.0),
                    "cooldown_remaining": max(0, a.cooldown_until - time.time()),
                    "consecutive_errors": a.consecutive_errors,
                }
                for a in self.accounts
            ],
        }
    
    def reload_credentials(self):
        """Reload credential files (e.g., after token refresh)."""
        for account in self.accounts:
            try:
                with open(account.credential_path) as f:
                    creds = json.load(f)
                account.credentials = creds
                logger.info(f"Reloaded credentials for {account.email}")
            except Exception as e:
                logger.error(f"Failed to reload credentials for {account.email}: {e}")


# Global singleton
_rotator: Optional[AccountRotator] = None


def get_rotator() -> AccountRotator:
    """Get or create the global account rotator."""
    global _rotator
    if _rotator is None:
        _rotator = AccountRotator()
    return _rotator
