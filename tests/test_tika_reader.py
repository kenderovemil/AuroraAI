import unittest
from unittest.mock import patch
import os
import importlib.util

# Load the tika_reader module directly by file path to avoid package import issues
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
module_path = os.path.join(repo_root, 'tools', 'apache_tika', 'tika_reader.py')
spec = importlib.util.spec_from_file_location('tika_reader', module_path)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)  # type: ignore


class TestTikaReaderHybrid(unittest.TestCase):
    def test_preview_text(self):
        self.assertEqual(tr.preview_text('', 10), '')
        self.assertEqual(tr.preview_text('hello', 10), 'hello')
        self.assertEqual(tr.preview_text('abcdefghijklmnopqrstuvwxyz', 5), 'abcde')

    def test_tika_sufficient(self):
        with patch.object(tr, 'extract_text_tika', return_value=('x' * 60000, {'a': 'b'})) as t:
            text, meta = tr.extract_text_hybrid('dummy.pdf', min_length=50000)
            self.assertEqual(text, 'x' * 60000)
            self.assertEqual(meta, {'a': 'b'})

    def test_tika_short_fitz_longer(self):
        with patch.object(tr, 'extract_text_tika', return_value=('short', {'m': 'd'})):
            with patch.object(tr, 'extract_text_fitz', return_value='f' * 20000):
                text, meta = tr.extract_text_hybrid('dummy.pdf', min_length=50000)
                self.assertEqual(text, 'f' * 20000)
                self.assertEqual(meta, {'m': 'd'})

    def test_tika_and_fitz_short_ocr_longer(self):
        with patch.object(tr, 'extract_text_tika', return_value=('short', {})):
            with patch.object(tr, 'extract_text_fitz', return_value='short2'):
                with patch.object(tr, 'extract_text_ocr', return_value='o' * 15000):
                    text, meta = tr.extract_text_hybrid('dummy.pdf', min_length=50000)
                    self.assertEqual(text, 'o' * 15000)

    def test_force_ocr(self):
        with patch.object(tr, 'extract_text_tika', return_value=('longtext', {})):
            with patch.object(tr, 'extract_text_fitz', return_value='longtext'):
                with patch.object(tr, 'extract_text_ocr', return_value='ocrtext'):
                    text, meta = tr.extract_text_hybrid('dummy.pdf', min_length=1, ocr=True, ocr_max_pages=1)
                    self.assertEqual(text, 'ocrtext')


if __name__ == '__main__':
    unittest.main()
