"""Duplicate-filter compatibility shim for scrapy-splash requests."""

from __future__ import annotations

from scrapy_splash import SplashAwareDupeFilter


class PatchedSplashAwareDupeFilter(SplashAwareDupeFilter):
    """Respect Scrapy's ``dont_filter`` flag before Splash fingerprinting."""

    def request_seen(self, request):
        if request.meta.get("dont_filter"):
            return False
        return super().request_seen(request)
