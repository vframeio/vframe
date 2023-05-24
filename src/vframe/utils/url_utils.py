#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

from urllib import request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, fp: str) -> None:
    """Downloads URL to filepath
    :param url: url to remote file
    :param fp: filepath to save as local file
    """
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        request.urlretrieve(url, filename=fp, reporthook=t.update_to)
