import zipfile
import tarfile

from tqdm import tqdm

from .logger import LOGGER


def extract_progress(compressed_obj):
    LOGGER.info("Extracting dataset")
    if isinstance(compressed_obj, tarfile.TarFile):
        iterable = compressed_obj
        length = len(compressed_obj.getmembers())
    elif isinstance(compressed_obj, zipfile.ZipFile):
        iterable = compressed_obj.namelist()
        length = len(iterable)
    for member in tqdm.tqdm(iterable, total=length):
        yield member
