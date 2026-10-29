"""Consent-gated, redistributable procedure sources; no fetched images or page scripts."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import httpx

MAX_SOURCE_BYTES = 8 * 1024 * 1024


class ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'footer', 'noscript', 'svg'):
            self.hidden += 1
        if tag in ('p', 'div', 'li', 'br', 'h1', 'h2', 'h3') and not self.hidden:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'footer', 'noscript', 'svg'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def guard_source(ref: str, licence: str, attribution: str, privacy_confirmed: bool):
    if not privacy_confirmed:
        raise ValueError('Confirm that the source contains no private/patient information with --privacy-confirmed')
    if 'ifixit' in ref.lower() or 'ifixit' in attribution.lower():
        raise ValueError('iFixit sources cannot be recorded in redistributable authoring packs')
    if not re.fullmatch(r'original|CC BY (?:3\.0|4\.0)', licence):
        raise ValueError('Source licence must be original, CC BY 3.0 or CC BY 4.0 (not NC/SA)')
    if not attribution.strip():
        raise ValueError('Source attribution is required, including author/source and changes for CC BY')


@dataclass(frozen=True)
class Source:
    metadata: dict
    text: str
    pdf: bytes | None = None


def load_source(ref: str, *, licence: str, attribution: str, privacy_confirmed: bool,
                transport: httpx.BaseTransport | None = None) -> Source:
    guard_source(ref, licence, attribution, privacy_confirmed)
    url = urlsplit(ref)
    remote = url.scheme in ('http', 'https')
    kind = 'url' if remote else 'text'
    if remote:
        with httpx.Client(transport=transport, timeout=30, follow_redirects=False, trust_env=False) as client:
            target = ref
            for _ in range(6):
                guard_source(target, licence, attribution, privacy_confirmed)
                with client.stream('GET', target) as response:
                    if response.is_redirect:
                        target = str(response.url.join(response.headers['location']))
                        continue
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_SOURCE_BYTES:
                            raise ValueError('source exceeds 8 MiB')
                    mime = response.headers.get('content-type', '')
                    raw = bytes(data)
                    break
            else:
                raise ValueError('too many source redirects')
    else:
        path = Path(ref)
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError('source exceeds 8 MiB')
        raw, mime = path.read_bytes(), ''
    pdf = raw if raw.startswith(b'%PDF-') else None
    if pdf:
        kind = 'pdf'
        try:
            result = subprocess.run(['pdftotext', '-layout', '-', '-'], input=pdf, capture_output=True, timeout=30, check=True)
        except FileNotFoundError:
            raise ValueError('PDF ingestion requires Poppler pdftotext') from None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f'PDF text extraction failed: {exc}') from None
        text = result.stdout.decode('utf-8')
    else:
        text = raw.decode('utf-8-sig')
        if 'html' in mime or re.search(r'<(?:html|body|article)\b', text, re.I):
            parser = ReadableHTML()
            parser.feed(text)
            text = ''.join(parser.parts)
    if re.search(r'(?:https?://)?(?:www\.)?ifixit\.com\b', text, re.I):
        raise ValueError('iFixit-linked material needs a separate non-redistributable workflow; do not record it here')
    if not text.strip():
        raise ValueError('No readable source text; OCR scanned PDFs before authoring')
    return Source({'kind': kind, 'ref': ref, 'licence': licence, 'attribution': attribution}, text.strip(), pdf)
