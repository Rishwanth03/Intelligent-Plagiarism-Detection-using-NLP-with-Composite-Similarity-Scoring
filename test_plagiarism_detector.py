import os
import unittest

from app.nlp.plagiarism_detector import PlagiarismDetector


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.detector = PlagiarismDetector()

    def test_lowercase(self):
        self.assertIn("hello", self.detector.preprocess_text("HELLO"))

    def test_remove_urls(self):
        output = self.detector.preprocess_text("visit https://example.com now")
        self.assertNotIn("https", output)

    def test_remove_emails(self):
        output = self.detector.preprocess_text("email me at abc@xyz.com")
        self.assertNotIn("abc", output)

    def test_remove_special_characters(self):
        output = self.detector.preprocess_text("hello!!! #### world")
        self.assertNotIn("!", output)

    def test_trim_spaces(self):
        output = self.detector.preprocess_text("   hello   world   ")
        self.assertTrue(output.startswith("hello"))

    def test_empty_text(self):
        output = self.detector.preprocess_text("")
        self.assertEqual(output, "")

    def test_none_text(self):
        output = self.detector.preprocess_text(None)
        self.assertEqual(output, "")

    def test_stopword_removal(self):
        output = self.detector.preprocess_text("the cat and the dog")
        self.assertNotIn(" the ", f" {output} ")

    def test_tokenization_basic(self):
        tokens = self.detector._tokenize("hello world")
        self.assertGreaterEqual(len(tokens), 2)

    def test_lemmatization_plural(self):
        output = self.detector.preprocess_text("cars")
        self.assertIn("car", output)


class TestSimilarityMethods(unittest.TestCase):
    def setUp(self):
        self.detector = PlagiarismDetector()

    def test_cosine_identical(self):
        score = self.detector.cosine_similarity("test text", "test text")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_cosine_different(self):
        score = self.detector.cosine_similarity("apple orange", "table chair")
        self.assertLess(score, 0.4)

    def test_cosine_empty(self):
        score = self.detector.cosine_similarity("", "")
        self.assertEqual(score, 0.0)

    def test_cosine_non_negative(self):
        score = self.detector.cosine_similarity("one", "two")
        self.assertGreaterEqual(score, 0.0)

    def test_cosine_not_above_one(self):
        score = self.detector.cosine_similarity("one two", "one two")
        self.assertLessEqual(score, 1.0)

    def test_ngram_identical(self):
        score = self.detector.ngram_similarity("a b c d", "a b c d")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_ngram_partial(self):
        score = self.detector.ngram_similarity("a b c d", "a b x y")
        self.assertGreater(score, 0.0)

    def test_ngram_empty(self):
        score = self.detector.ngram_similarity("", "")
        self.assertEqual(score, 0.0)

    def test_ngram_range(self):
        score = self.detector.ngram_similarity("a b c", "d e f")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_lexical_identical(self):
        score = self.detector.lexical_overlap("a b c", "a b c")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_lexical_partial(self):
        score = self.detector.lexical_overlap("a b c", "a x y")
        self.assertGreater(score, 0.0)

    def test_lexical_empty(self):
        score = self.detector.lexical_overlap("", "")
        self.assertEqual(score, 0.0)

    def test_lexical_zero_division_guard(self):
        score = self.detector.lexical_overlap("", "abc")
        self.assertEqual(score, 0.0)

    def test_lexical_range(self):
        score = self.detector.lexical_overlap("x y", "x z")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_generate_ngrams_bigram(self):
        ngrams = self.detector._generate_ngrams("a b c", 2)
        self.assertIn(("a", "b"), ngrams)

    def test_generate_ngrams_too_short(self):
        ngrams = self.detector._generate_ngrams("a", 2)
        self.assertEqual(len(ngrams), 0)


class TestCompositeAndClassification(unittest.TestCase):
    def setUp(self):
        self.detector = PlagiarismDetector()

    def test_composite_formula_bounds(self):
        score = self.detector.composite_score("same text", "same text")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_detect_contains_all_fields(self):
        result = self.detector.detect("a b c", "a b x")
        for key in [
            "cosine_similarity",
            "ngram_similarity",
            "lexical_overlap",
            "composite_score",
            "classification",
        ]:
            self.assertIn(key, result)

    def test_classify_no_plagiarism(self):
        self.assertEqual(self.detector.classify_score(0.05), "No Plagiarism")

    def test_classify_minor_at_boundary(self):
        self.assertEqual(self.detector.classify_score(0.10), "Minor")

    def test_classify_minor(self):
        self.assertEqual(self.detector.classify_score(0.25), "Minor")

    def test_classify_moderate_boundary(self):
        self.assertEqual(self.detector.classify_score(0.30), "Minor")

    def test_classify_moderate(self):
        self.assertEqual(self.detector.classify_score(0.45), "Moderate")

    def test_classify_severe_boundary(self):
        self.assertEqual(self.detector.classify_score(0.61), "Severe")

    def test_classify_severe_high(self):
        self.assertEqual(self.detector.classify_score(0.95), "Severe")

    def test_detect_classification_type(self):
        result = self.detector.detect("abc def", "abc def")
        self.assertIsInstance(result["classification"], str)

    def test_vectorizer_max_features(self):
        self.assertEqual(self.detector.vectorizer.max_features, 5000)

    def test_vectorizer_ngram_range(self):
        self.assertEqual(self.detector.vectorizer.ngram_range, (1, 2))

    def test_extract_tfidf_shape(self):
        matrix = self.detector.extract_tfidf_features(["a b", "a c"])
        self.assertEqual(matrix.shape[0], 2)

    def test_detect_on_empty_and_nonempty(self):
        result = self.detector.detect("", "text")
        self.assertGreaterEqual(result["composite_score"], 0.0)

    def test_detect_on_unicode_input(self):
        result = self.detector.detect("naive cafe", "naive cafe")
        self.assertGreaterEqual(result["composite_score"], 0.0)

    def test_default_tuned_profile_is_pairwise(self):
        detector = PlagiarismDetector()
        self.assertEqual(detector.tuned_profile, "pairwise")

    def test_tuned_profile_scoped_env_is_used(self):
        prev_profile = os.environ.get("TUNED_PROFILE")
        prev_weights_bbc = os.environ.get("TUNED_WEIGHTS_BBC")
        prev_threshold_bbc = os.environ.get("TUNED_THRESHOLD_BBC")
        try:
            os.environ["TUNED_PROFILE"] = "bbc"
            os.environ["TUNED_WEIGHTS_BBC"] = "0.5,0.3,0.2,0.0"
            os.environ["TUNED_THRESHOLD_BBC"] = "0.5"
            detector = PlagiarismDetector()
            self.assertEqual(detector.tuned_profile, "bbc")
            self.assertEqual(detector.tuned_threshold, 0.5)
            self.assertAlmostEqual(detector.tuned_weights[0], 0.5)
        finally:
            if prev_profile is None:
                os.environ.pop("TUNED_PROFILE", None)
            else:
                os.environ["TUNED_PROFILE"] = prev_profile
            if prev_weights_bbc is None:
                os.environ.pop("TUNED_WEIGHTS_BBC", None)
            else:
                os.environ["TUNED_WEIGHTS_BBC"] = prev_weights_bbc
            if prev_threshold_bbc is None:
                os.environ.pop("TUNED_THRESHOLD_BBC", None)
            else:
                os.environ["TUNED_THRESHOLD_BBC"] = prev_threshold_bbc


if __name__ == "__main__":
    unittest.main()
