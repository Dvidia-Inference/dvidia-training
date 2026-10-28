"""Bounded anonymous public readback for the unchanged grip-clock publisher.

This transport returns bytes only. The publisher still verifies every expected
SHA-256/Git/LFS identity, complete inventory, and exact parent. It never uploads,
uses the SDK credential client, or consults environment proxy credentials.
"""
from __future__ import annotations

import http.client
import re
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import (HTTPSHandler, HTTPRedirectHandler, ProxyHandler,
                            Request, build_opener)

import publish_grip_clock_v03 as pub
from publish import PublicationError


MAX_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (.25, .5, 1.)
RETRY_STATUS = frozenset((408, 425, 429, 500, 502, 503, 504))
HEADERS = {'User-Agent': 'DVIDIA-GripClock-PublicReadback/0.3',
           'Accept-Encoding': 'identity'}
ROOTS = {'model': '', 'dataset': 'datasets/', 'space': 'spaces/'}


class PublicReadbackError(PublicationError):
    """Safe diagnostics only: never retain HTTP exception URLs, bodies or headers."""
    def __init__(self, path, failure_class, http_status, attempts, check_code=None):
        super().__init__('Anonymous public readback failed; inspect the safe transport diagnostic.')
        self.path, self.failure_class = path, failure_class
        self.http_status, self.attempts, self.check_code = http_status, attempts, check_code


CHECK_CODES = {
    'Anonymous public readback differs from its declared content length.': 'declared_length_mismatch',
    'Anonymous public readback requires identity content encoding.': 'nonidentity_encoding',
    'Anonymous public readback exceeds its bounded byte limit.': 'body_oversize',
    'Anonymous public readback exceeds its declared byte limit.': 'declared_oversize',
    'Anonymous readback has multiple declared content lengths.': 'duplicate_content_length',
    'Anonymous readback has an invalid declared content length.': 'invalid_content_length',
    'Anonymous readback redirects must stay on official HF HTTPS hosts.': 'foreign_redirect',
    'Anonymous public readback exceeded its redirect bound.': 'redirect_limit',
}


def _safe_url(url):
    """Allow signed public HF/CDN redirects, never foreign destinations."""
    if (not isinstance(url, str) or not url or not url.isascii()
            or any(ord(char) <= 32 or ord(char) == 127 for char in url)):
        raise PublicationError('Invalid anonymous public HTTPS destination.')
    try:
        parts = urlsplit(url)
        host = parts.hostname
        forbidden = (parts.scheme != 'https' or parts.username is not None
                     or parts.password is not None or parts.port is not None
                     or parts.fragment or not host)
    except ValueError:
        raise PublicationError('Invalid anonymous public HTTPS destination.') from None
    if (forbidden or not any(host == domain or host.endswith('.' + domain)
                             for domain in ('huggingface.co', 'hf.co'))):
        raise PublicationError('Anonymous readback redirects must stay on official HF HTTPS hosts.')
    return url


def _public_url(repo, name, kind, revision):
    if (kind not in ROOTS or repo != pub.REPOS[kind]
            or not isinstance(revision, str) or pub.OID.fullmatch(revision) is None):
        raise PublicationError('Anonymous readback requires a fixed public repository and exact revision.')
    pub.relative(name, attributes=True)
    return _safe_url('https://huggingface.co/' + ROOTS[kind] + repo + '/resolve/'
                     + revision + '/' + quote(name, safe='/') + '?download=true')


class _PublicRedirect(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, request, response, code, message, headers, newurl):
        _safe_url(newurl)
        if code not in (301, 302, 303, 307, 308) or request.get_method() != 'GET':
            raise PublicationError('Unexpected anonymous public redirect method or status.')
        # Reconstruct the fixed headers rather than forwarding server/caller state.
        return Request(newurl, headers=dict(HEADERS), method='GET')

    def http_error_302(self, request, response, code, message, headers):
        # urllib's default redirect handler drains the redirect body without a
        # byte bound. Close it unread and follow only a validated destination.
        try:
            location = headers.get('Location') or headers.get('URI')
            if (not isinstance(location, str) or not location or not location.isascii()
                    or any(ord(char) <= 32 or ord(char) == 127 for char in location)):
                raise PublicationError('Invalid anonymous public redirect destination.')
            following = self.redirect_request(request, response, code, message, headers,
                                              urljoin(request.full_url, location))
            count = getattr(request, '_dvidia_redirects', 0) + 1
            if count > self.max_redirections:
                raise PublicationError('Anonymous public readback exceeded its redirect bound.')
            following._dvidia_redirects = count
        finally:
            response.close()
        return self.parent.open(following, timeout=request.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _opener():
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), _PublicRedirect())
    opener.addheaders = []
    return opener


def _body(response):
    if response.status != 200:
        raise PublicationError('Anonymous public readback requires HTTP 200.')
    _safe_url(response.geturl())
    encoding = response.headers.get('Content-Encoding', 'identity').strip().lower()
    if encoding != 'identity':
        raise PublicationError('Anonymous public readback requires identity content encoding.')
    lengths = (response.headers.get_all('Content-Length')
               if hasattr(response.headers, 'get_all') else
               ([response.headers['Content-Length']] if 'Content-Length' in response.headers else []))
    if lengths is None:
        lengths = []
    if len(lengths) > 1:
        raise PublicationError('Anonymous readback has multiple declared content lengths.')
    declared = None
    if lengths:
        value = lengths[0].strip()
        if re.fullmatch(r'[0-9]+', value) is None:
            raise PublicationError('Anonymous readback has an invalid declared content length.')
        significant = value.lstrip('0') or '0'
        if len(significant) > len(str(MAX_BYTES)):
            raise PublicationError('Anonymous public readback exceeds its declared byte limit.')
        declared = int(significant)
        if declared > MAX_BYTES:
            raise PublicationError('Anonymous public readback exceeds its declared byte limit.')
    raw = response.read(MAX_BYTES + 1)
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise PublicationError('Anonymous public readback exceeds its bounded byte limit.')
    if declared is not None and len(raw) != declared:
        raise PublicationError('Anonymous public readback differs from its declared content length.')
    return raw


def download(repo, name, kind, revision):
    """GET at most four times; no authentication, cookies, proxies, or writes."""
    url = _public_url(repo, name, kind, revision)
    opener = _opener()
    last_status, failure_class = None, None
    for attempt in range(MAX_ATTEMPTS):
        try:
            request = Request(url, headers=dict(HEADERS), method='GET')
            with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
                last_status = response.status
                return _body(response)
        except PublicationError as error:
            raise PublicReadbackError(name, type(error).__name__, last_status, attempt+1,
                                      CHECK_CODES.get(str(error), 'public_response_check')) from None
        except HTTPError as error:
            code = error.code
            last_status, failure_class = code, type(error).__name__
            error.close()
            if code not in RETRY_STATUS:
                raise PublicReadbackError(name, failure_class, code, attempt+1,
                                          'unexpected_http_status') from None
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            reason = getattr(error, 'reason', error)
            failure_class = type(reason).__name__
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise PublicReadbackError(name, failure_class, last_status, attempt+1,
                                          'tls_verification_failed') from None
        if attempt == MAX_ATTEMPTS - 1:
            raise PublicReadbackError(name, failure_class, last_status, attempt+1,
                                      'attempts_exhausted') from None
        time.sleep(BACKOFF_SECONDS[attempt])
    raise PublicationError('Anonymous public readback did not return bytes.')
