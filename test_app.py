import io
import unittest

import app as sentiment_app


class FakeVectorizer:
    def transform(self, values):
        return values


class FakeModel:
    classes_ = [0, 1]

    def predict(self, values):
        return [1 if "great" in values[0] or "fast" in values[0] else 0]

    def predict_proba(self, values):
        if "great" in values[0] or "fast" in values[0]:
            return [[0.1, 0.9]]
        return [[0.8, 0.2]]


class SentimentAppTests(unittest.TestCase):
    def setUp(self):
        sentiment_app.vectorizer = FakeVectorizer()
        sentiment_app.model = FakeModel()
        sentiment_app.app.config["TESTING"] = True
        self.client = sentiment_app.app.test_client()

    def test_single_api_accepts_xquik_tweet_text_alias(self):
        response = self.client.post(
            "/api/sentiment",
            json={"Tweet Text": "Great support from the team"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["sentiment"], "Positive")
        self.assertEqual(payload["confidence"], 90.0)

    def test_confidence_uses_model_class_order(self):
        class SignedLabelModel:
            classes_ = [-1, 1]

            def predict(self, values):
                return [-1]

            def predict_proba(self, values):
                return [[0.8, 0.2]]

        sentiment_app.model = SignedLabelModel()

        prediction = sentiment_app.predict_feedback("Slow checkout")

        self.assertEqual(prediction["sentiment"], "Negative")
        self.assertEqual(prediction["confidence"], 80.0)

    def test_batch_api_filters_empty_and_duplicate_csv_rows(self):
        csv_file = io.BytesIO(
            b"Tweet Text,Tweet Created At\n"
            b"Fast service,2026-07-05\n"
            b",2026-07-05\n"
            b"Fast service,2026-07-05\n"
            b"Slow checkout,2026-07-05\n"
        )
        response = self.client.post(
            "/api/sentiment/batch",
            data={"file": (csv_file, "xquik.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            [row["feedback"] for row in payload["results"]],
            ["Fast service", "Slow checkout"],
        )

    def test_batch_api_reports_missing_text_columns(self):
        response = self.client.post(
            "/api/sentiment/batch",
            json={"rows": [{"score": 1}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("text columns", response.get_json()["error"])

    def test_batch_api_rejects_malformed_csv(self):
        response = self.client.post(
            "/api/sentiment/batch",
            data={"file": (io.BytesIO(b'Tweet Text\n"unterminated'), "broken.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"],
            "Upload a valid UTF-8 CSV file.",
        )

    def test_batch_api_requires_array_of_objects(self):
        response = self.client.post(
            "/api/sentiment/batch",
            json={"rows": "not-an-array"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"],
            "JSON rows must be an array of objects.",
        )

    def test_web_form_handles_blank_feedback(self):
        response = self.client.post("/", data={"feedback": "   "})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Feedback text is required.", response.data)


if __name__ == "__main__":
    unittest.main()
