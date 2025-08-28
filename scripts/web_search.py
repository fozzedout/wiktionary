#!/usr/bin/env python3
"""
Web search functionality for the small dictionary builder.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    # Use stdlib if requests isn't available.
    import urllib.parse
    import urllib.request as _urllib
except Exception:  # pragma: no cover
    _urllib = None
    urllib = None


def _search_web_for_context(term: str, pos: Optional[str], search_engines: Optional[List[str]] = None) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Search for additional context about a term using multiple web sources.

    Returns (context_string, search_results_list) where search_results_list contains
    detailed information about each search result for storage in database.
    """
    if search_engines is None:
        search_engines = ["duckduckgo", "wikipedia"]

    if _urllib is None:
        return None, []

    context_parts = []
    search_results = []

    try:
        # Construct search query
        query = f"{term} definition meaning"
        if pos:
            query += f" {pos}"

        # 1. DuckDuckGo Instant Answer API (no API key required)
        if "duckduckgo" in search_engines and _urllib is not None:
            try:
                import urllib.parse
                encoded_query = urllib.parse.quote(query)
                url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1"

                req = _urllib.Request(url, headers={"User-Agent": "Dictionary-Builder/1.0"})
                with _urllib.urlopen(req, timeout=10.0) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                    # Abstract (main definition)
                    if data.get("Abstract"):
                        context_parts.append(data["Abstract"])
                        search_results.append({
                            "source": "duckduckgo_abstract",
                            "title": data.get("Heading", term),
                            "snippet": data["Abstract"],
                            "url": data.get("AbstractURL", ""),
                            "rank": 1
                        })

                    # Related topics
                    if data.get("RelatedTopics"):
                        for i, topic in enumerate(data["RelatedTopics"][:3]):  # Limit to first 3
                            if isinstance(topic, dict) and topic.get("Text"):
                                context_parts.append(topic["Text"])
                                search_results.append({
                                    "source": "duckduckgo_related",
                                    "title": topic.get("FirstURL", "").split('/')[-1] if topic.get("FirstURL") else f"Related Topic {i+1}",
                                    "snippet": topic["Text"],
                                    "url": topic.get("FirstURL", ""),
                                    "rank": i + 2
                                })

                    # Answer (if available)
                    if data.get("Answer"):
                        context_parts.append(data["Answer"])
                        search_results.append({
                            "source": "duckduckgo_answer",
                            "title": f"{term} - Answer",
                            "snippet": data["Answer"],
                            "url": data.get("AnswerURL", ""),
                            "rank": 1
                        })

            except Exception as e:
                # Continue with other search engines if DuckDuckGo fails
                pass

        # 2. Wikipedia search
        if "wikipedia" in search_engines and _urllib is not None:
            try:
                # Search Wikipedia API
                import urllib.parse
                search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"

                req = _urllib.Request(search_url, headers={"User-Agent": "Dictionary-Builder/1.0"})
                with _urllib.urlopen(req, timeout=10.0) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                    if data.get("extract"):
                        # Get first few sentences from Wikipedia extract
                        extract = data["extract"]
                        # Limit to reasonable length
                        parts = re.split(r"(?<=[.!?])\s+", extract.strip())
                        wiki_snippet = " ".join(parts[:2]) if parts else extract
                        context_parts.append(wiki_snippet)
                        search_results.append({
                            "source": "wikipedia",
                            "title": data.get("title", term),
                            "snippet": wiki_snippet,
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "rank": 1
                        })

            except Exception as e:
                # Continue if Wikipedia fails
                pass

        # 3. Google Custom Search API (if API key available)
        if "google" in search_engines:
            try:
                # This would require a Google Custom Search API key
                # For now, we'll skip this as it requires API keys
                pass
            except Exception:
                pass

        if context_parts:
            return " | ".join(context_parts), search_results

    except Exception:
        pass  # Fail silently and continue without web context

    return None, []