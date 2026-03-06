
import warnings
import requests
import urllib

from typing import Any, Dict, Iterator, Optional, Tuple
from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper


# TODO(642): Remove this helper function when https://bugs.python.org/issue42627 is resolved
def _get_proxies() -> Optional[Dict[str, str]]:
    import os

    if os.name == "nt":
        proxies = urllib.request.getproxies()
        address = proxies.get("https")
        # The default proxy type of Windows is HTTP
        if address and address.startswith("https"):
            address = "http" + address[5:]
            proxies["https"] = address
            return proxies
    return None


def _get_response_from_http(
    url: str, *, timeout: Optional[float], **query_params: Optional[Dict[str, Any]]
) -> Tuple[str, StreamWrapper]:
    with requests.Session() as session:
        proxies = _get_proxies()
        r = session.get(url, timeout=timeout, proxies=proxies, stream=True, **query_params)  # type: ignore[arg-type]
    r.raise_for_status()
    return url, StreamWrapper(r.raw)


@functional_datapipe("read_from_http")
class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes file URLs (HTTP URLs pointing to files), and yields tuples of file URL and
    IO stream (functional name: ``read_from_http``).

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout: timeout in seconds for HTTP request
        skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
        **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/

    Example:

    .. testcode::

        from touchtext.datapipes.iter import IterableWrapper, HttpReader

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        query_params = {"auth" : ("fake_username", "fake_password"), "allow_redirects" : True}
        timeout = 120
        http_reader_dp = HttpReader(IterableWrapper([file_url]), timeout=timeout, **query_params)
        reader_dp = http_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        print((path, line))

    Output:

    .. testoutput::

        ('https://raw.githubusercontent.com/pytorch/data/main/LICENSE', b'BSD 3-Clause License')
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        timeout: Optional[float] = None,
        skip_on_error: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.timeout = timeout
        self.skip_on_error = skip_on_error
        self.query_params = kwargs

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            try:
                yield _get_response_from_http(url, timeout=self.timeout, **self.query_params)
            except Exception as e:
                if self.skip_on_error:
                    warnings.warn(f"{e}, skipping...")
                else:
                    raise

    def __len__(self) -> int:
        return len(self.source_datapipe)
