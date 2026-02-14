"""
Rotated Client — Multi-account request routing.

Wraps the Google API client to route requests through the account rotator.
Each request gets fresh credentials from the next available account.
"""

import json
import logging
import requests as http_requests
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

from .account_rotator import get_rotator, AccountState
from .utils import get_user_agent, get_client_metadata
from .config import (
    CODE_ASSIST_ENDPOINT, SCOPES,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name, is_search_model,
    get_thinking_budget, should_include_thoughts,
)
import asyncio

logger = logging.getLogger(__name__)

# Cache onboarding status per account
_onboarded_accounts: set[str] = set()


def _get_google_credentials(account: AccountState) -> Credentials | None:
    """Create and refresh Google Credentials from account data."""
    creds_data = account.credentials
    
    try:
        creds = Credentials.from_authorized_user_info(creds_data, SCOPES)
    except Exception as e:
        logger.error(f"Failed to create credentials for {account.email}: {e}")
        return None
    
    # Refresh if expired
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            # Update stored token for next time
            account.credentials["token"] = creds.token
            if creds.expiry:
                from datetime import timezone
                if creds.expiry.tzinfo is None:
                    expiry_utc = creds.expiry.replace(tzinfo=timezone.utc)
                else:
                    expiry_utc = creds.expiry
                account.credentials["expiry"] = expiry_utc.isoformat()
            logger.debug(f"Refreshed credentials for {account.email}")
        except Exception as e:
            logger.error(f"Failed to refresh credentials for {account.email}: {e}")
            return None
    elif not creds.token:
        logger.error(f"No access token for {account.email}")
        return None
    
    return creds


def _ensure_onboarded(creds: Credentials, project_id: str, account_email: str) -> bool:
    """Ensure the account is onboarded with Code Assist."""
    if account_email in _onboarded_accounts:
        return True
    
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    
    load_payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }
    
    try:
        resp = http_requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(load_payload),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("currentTier"):
            _onboarded_accounts.add(account_email)
            return True
        
        # Need to onboard
        tier = None
        for allowed_tier in data.get("allowedTiers", []):
            if allowed_tier.get("isDefault"):
                tier = allowed_tier
                break
        
        if not tier:
            tier = {
                "name": "", "description": "",
                "id": "legacy-tier",
                "userDefinedCloudaicompanionProject": True,
            }
        
        onboard_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(project_id),
        }
        
        onboard_resp = http_requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            data=json.dumps(onboard_payload),
            headers=headers,
            timeout=30,
        )
        onboard_resp.raise_for_status()
        _onboarded_accounts.add(account_email)
        return True
        
    except Exception as e:
        logger.error(f"Onboarding failed for {account_email}: {e}")
        return False


def send_rotated_request(payload: dict, is_streaming: bool = False) -> Response:
    """
    Send a request using the next available account from the rotation pool.
    Falls back to single-account mode if no rotator accounts are available.
    """
    rotator = get_rotator()
    
    # If no accounts loaded, fall back to original single-account path
    if not rotator.accounts:
        logger.warning("No rotator accounts — falling back to single-account mode")
        from .google_api_client import send_gemini_request
        return send_gemini_request(payload, is_streaming)
    
    account = rotator.get_next_account()
    if account is None:
        return Response(
            content=json.dumps({
                "error": {
                    "message": "All accounts exhausted or in cooldown. Try again later.",
                    "type": "rate_limit_error",
                    "code": 429
                }
            }),
            status_code=429,
            media_type="application/json",
        )
    
    logger.info(f"Routing request to {account.email} (RPM: {account.requests_in_window()}/{rotator.max_rpm})")
    
    # Get credentials
    creds = _get_google_credentials(account)
    if not creds:
        rotator.record_error(account, 500, "Failed to get credentials")
        # Try next account
        account2 = rotator.get_next_account()
        if account2:
            creds = _get_google_credentials(account2)
            account = account2
        if not creds:
            return Response(
                content=json.dumps({"error": {"message": "Authentication failed for all accounts"}}),
                status_code=500,
                media_type="application/json",
            )
    
    # Ensure onboarded
    project_id = account.project_id
    if not _ensure_onboarded(creds, project_id, account.email):
        rotator.record_error(account, 403, "Onboarding failed")
        return Response(
            content=json.dumps({"error": {"message": f"Onboarding failed for {account.email}"}}),
            status_code=500,
            media_type="application/json",
        )
    
    # Build final payload
    final_payload = {
        "model": payload.get("model"),
        "project": project_id,
        "request": payload.get("request", {}),
    }
    
    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"
    
    request_headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    
    # Send the request
    try:
        if is_streaming:
            resp = http_requests.post(
                target_url,
                data=json.dumps(final_payload),
                headers=request_headers,
                stream=True,
            )
            if resp.status_code == 200:
                rotator.record_success(account)
            else:
                rotator.record_error(account, resp.status_code, f"HTTP {resp.status_code}")
            return _handle_streaming_response(resp)
        else:
            resp = http_requests.post(
                target_url,
                data=json.dumps(final_payload),
                headers=request_headers,
            )
            if resp.status_code == 200:
                rotator.record_success(account)
            else:
                rotator.record_error(account, resp.status_code, f"HTTP {resp.status_code}")
            return _handle_non_streaming_response(resp)
            
    except http_requests.exceptions.RequestException as e:
        rotator.record_error(account, 502, str(e))
        return Response(
            content=json.dumps({"error": {"message": f"Request failed: {e}"}}),
            status_code=502,
            media_type="application/json",
        )


# ---- Response handlers (same as google_api_client.py but self-contained) ----

def _handle_streaming_response(resp) -> StreamingResponse:
    """Handle streaming response from Google API."""
    if resp.status_code != 200:
        error_message = f"Google API error: {resp.status_code}"
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except Exception:
            pass
        
        async def error_generator():
            error_response = {
                "error": {"message": error_message, "type": "api_error", "code": resp.status_code}
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8')
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )
    
    async def stream_generator():
        try:
            with resp:
                for chunk in resp.iter_lines():
                    if chunk:
                        if not isinstance(chunk, str):
                            chunk = chunk.decode('utf-8', 'ignore')
                        if chunk.startswith('data: '):
                            chunk = chunk[len('data: '):]
                            try:
                                obj = json.loads(chunk)
                                if "response" in obj:
                                    response_chunk = obj["response"]
                                    yield f"data: {json.dumps(response_chunk, separators=(',', ':'))}\n\n".encode('utf-8')
                                else:
                                    yield f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode('utf-8')
                                await asyncio.sleep(0)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f'data: {json.dumps({"error": {"message": str(e), "code": 500}})}\n\n'.encode('utf-8')
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


def _handle_non_streaming_response(resp) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            text = resp.text
            if text.startswith('data: '):
                text = text[len('data: '):]
            parsed = json.loads(text)
            content = parsed.get("response", parsed)
            return Response(
                content=json.dumps(content),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except (json.JSONDecodeError, AttributeError):
            return Response(content=resp.content, status_code=resp.status_code)
    else:
        try:
            error_data = resp.json()
            if "error" in error_data:
                return Response(
                    content=json.dumps({"error": error_data["error"]}),
                    status_code=resp.status_code,
                    media_type="application/json",
                )
        except Exception:
            pass
        return Response(content=resp.content, status_code=resp.status_code)
