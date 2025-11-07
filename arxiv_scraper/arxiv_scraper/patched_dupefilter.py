from scrapy_splash.dupefilter import SplashAwareDupeFilter
from scrapy.utils.request import fingerprint as new_fingerprint

class PatchedSplashAwareDupeFilter(SplashAwareDupeFilter):
    def __init__(self, *args, include_headers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_headers = include_headers or []

    def request_fingerprint(self, request):
        return new_fingerprint(request, include_headers=self.include_headers)
