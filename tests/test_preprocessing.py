"""
Unit tests for ReviewSense AI Preprocessing Pipeline

Run via:
    python3 -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path so predict module can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import (
    clean_text,
    tokenize_and_encode,
    pad_sequence,
    preprocess_review,
    START_INDEX,
    UNK_INDEX,
    PAD_INDEX
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample mini word index dictionary (word -> rank)
        self.sample_word_index = {
            "movie": 1,
            "film": 2,
            "great": 3,
            "terrible": 4,
            "acting": 5,
            "good": 6,
            "bad": 7
        }

    def test_clean_text_html_removal(self):
        raw = "This was a <b>terrific</b> movie!<br /><br />Loved it."
        expected = "this was a terrific movie loved it"
        self.assertEqual(clean_text(raw), expected)

    def test_clean_text_punctuation_and_casing(self):
        raw = "GREAT movie, 10/10! Highly recommended... must-see?!"
        cleaned = clean_text(raw)
        # Should have lowercased words without numbers or punctuation
        self.assertEqual(cleaned, "great movie highly recommended must see")

    def test_clean_text_empty_and_whitespace(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t  "), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(12345), "")

    def test_tokenize_and_encode_known_and_unknown_words(self):
        text = "great movie bad unknownword"
        # INDEX_OFFSET is 3:
        # 'great' rank 3 -> 3 + 3 = 6
        # 'movie' rank 1 -> 1 + 3 = 4
        # 'bad' rank 7 -> 7 + 3 = 10
        # 'unknownword' not in dict -> UNK_INDEX (2)
        tokens = tokenize_and_encode(text, self.sample_word_index, max_words=1000)
        expected = [START_INDEX, 6, 4, 10, UNK_INDEX]
        self.assertEqual(tokens, expected)

    def test_tokenize_vocab_cutoff(self):
        # If max_words is 8, words with rank + 3 >= 8 should map to UNK
        # 'acting' rank is 5 -> 5 + 3 = 8 >= 8 -> UNK
        text = "acting"
        tokens = tokenize_and_encode(text, self.sample_word_index, max_words=8)
        self.assertEqual(tokens, [START_INDEX, UNK_INDEX])

    def test_pad_sequence_shorter_than_maxlen(self):
        seq = [START_INDEX, 5, 8]
        padded = pad_sequence(seq, maxlen=6, pad_value=PAD_INDEX)
        # Should pre-pad with zeros
        self.assertEqual(padded, [0, 0, 0, START_INDEX, 5, 8])
        self.assertEqual(len(padded), 6)

    def test_pad_sequence_longer_than_maxlen(self):
        seq = [1, 2, 3, 4, 5, 6, 7, 8]
        padded = pad_sequence(seq, maxlen=5)
        # Should pre-truncate (keep last 5 items)
        self.assertEqual(padded, [4, 5, 6, 7, 8])
        self.assertEqual(len(padded), 5)

    def test_pad_sequence_exact_length(self):
        seq = [1, 2, 3, 4]
        padded = pad_sequence(seq, maxlen=4)
        self.assertEqual(padded, seq)

    def test_preprocess_review_pipeline(self):
        text = "The <br>GREAT</br> movie was acting!"
        maxlen = 10
        processed = preprocess_review(text, self.sample_word_index, maxlen=maxlen, max_words=100)
        self.assertEqual(len(processed), maxlen)
        # Check that the tokens end with valid IDs
        self.assertIn(START_INDEX, processed)


if __name__ == "__main__":
    unittest.main()
