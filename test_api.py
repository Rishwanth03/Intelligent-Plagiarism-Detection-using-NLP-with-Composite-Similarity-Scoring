import os
import unittest

from app import create_app
from app.services.container import detection_service, document_service


class TestAPI(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()
        document_service._documents.clear()
        detection_service._results.clear()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("services", payload)
        self.assertIn("tuned_config", payload["services"]["similarity"])
        self.assertIn("profile", payload["services"]["similarity"]["tuned_config"])

    def test_create_document_success(self):
        response = self.client.post("/api/documents", json={"text": "hello"})
        self.assertEqual(response.status_code, 201)

    def test_create_document_invalid(self):
        response = self.client.post("/api/documents", json={"text": ""})
        self.assertEqual(response.status_code, 400)

    def test_list_documents(self):
        self.client.post("/api/documents", json={"text": "doc1"})
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        self.assertIn("documents", response.get_json())

    def test_delete_document_not_found(self):
        response = self.client.delete("/api/documents/not-found")
        self.assertEqual(response.status_code, 404)

    def test_similarity_cosine(self):
        response = self.client.post(
            "/api/similarity/cosine",
            json={"text1": "a b c", "text2": "a b c"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("cosine_similarity", response.get_json())

    def test_similarity_ngram(self):
        response = self.client.post(
            "/api/similarity/ngram",
            json={"text1": "a b c", "text2": "a b d"},
        )
        self.assertEqual(response.status_code, 200)

    def test_similarity_lexical(self):
        response = self.client.post(
            "/api/similarity/lexical",
            json={"text1": "a b c", "text2": "a x y"},
        )
        self.assertEqual(response.status_code, 200)

    def test_similarity_composite(self):
        response = self.client.post(
            "/api/similarity/composite",
            json={"text1": "a b c", "text2": "a b c"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("classification", response.get_json())

    def test_similarity_validation(self):
        response = self.client.post("/api/similarity/cosine", json={"text1": "a"})
        self.assertEqual(response.status_code, 400)

    def test_detect_text_mode(self):
        response = self.client.post(
            "/api/detect",
            json={"source_text": "a b c", "target_text": "a b c"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.get_json())

    def test_detect_id_mode(self):
        source = self.client.post("/api/documents", json={"text": "a b c"}).get_json()[
            "document"
        ]["id"]
        target = self.client.post("/api/documents", json={"text": "a b c"}).get_json()[
            "document"
        ]["id"]
        response = self.client.post(
            "/api/detect", json={"source_id": source, "target_id": target}
        )
        self.assertEqual(response.status_code, 201)

    def test_get_detection_not_found(self):
        response = self.client.get("/api/detect/abc")
        self.assertEqual(response.status_code, 404)

    def test_get_detection_found(self):
        source = self.client.post("/api/documents", json={"text": "a b c"}).get_json()[
            "document"
        ]["id"]
        target = self.client.post("/api/documents", json={"text": "a b c"}).get_json()[
            "document"
        ]["id"]
        created = self.client.post(
            "/api/detect", json={"source_id": source, "target_id": target}
        ).get_json()["result"]["id"]
        response = self.client.get(f"/api/detect/{created}")
        self.assertEqual(response.status_code, 200)

    def test_report_pairwise(self):
        self.client.post("/api/documents", json={"text": "a b c"})
        self.client.post("/api/documents", json={"text": "a b d"})
        response = self.client.get("/api/report/pairwise")
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.get_json())

    def test_report_statistics(self):
        response = self.client.get("/api/report/statistics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("total_detections", response.get_json())

    def test_delete_document_success(self):
        doc_id = self.client.post("/api/documents", json={"text": "delete me"}).get_json()[
            "document"
        ]["id"]
        response = self.client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(response.status_code, 200)

    def test_detect_invalid_payload(self):
        response = self.client.post("/api/detect", json={"source_text": ""})
        self.assertEqual(response.status_code, 400)

    def test_pairwise_empty(self):
        response = self.client.get("/api/report/pairwise")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["results"], [])

    def test_statistics_defaults(self):
        response = self.client.get("/api/report/statistics")
        payload = response.get_json()
        self.assertEqual(payload["precision"], 0.0)
        self.assertEqual(payload["recall"], 0.0)

    def test_rewrite_suggest_validation(self):
        response = self.client.post("/api/rewrite/suggest", json={"source_text": "", "target_text": "abc"})
        self.assertEqual(response.status_code, 400)

    def test_rewrite_suggest_response_shape(self):
        response = self.client.post(
            "/api/rewrite/suggest",
            json={
                "source_text": "The cat sat on the mat. It looked at the moon.",
                "target_text": "The cat sat on the mat. It looked at the moon.",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("should_rewrite", payload)
        self.assertIn("threshold", payload)

    def test_rewrite_execute_with_permission(self):
        response = self.client.post(
            "/api/rewrite/execute",
            json={
                "source_text": "Machine learning is useful for document analysis.",
                "target_text": "Machine learning is useful for document analysis.",
                "user_permission": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("rewritten", payload)
        self.assertIn("before_analysis", payload)
        self.assertIn("after_analysis", payload)
        self.assertIn("attempts", payload)
        self.assertIn("selected_attempt", payload)

    def test_rewrite_audit_log_roundtrip(self):
        add_response = self.client.post(
            "/api/rewrite/audit",
            json={
                "timestamp": "2026-04-03T12:00:00Z",
                "segment_id": "seg-1",
                "action": "accepted",
                "original_text": "old",
                "rewritten_text": "new",
            },
        )
        self.assertEqual(add_response.status_code, 201)

        list_response = self.client.get("/api/rewrite/audit")
        self.assertEqual(list_response.status_code, 200)
        entries = list_response.get_json()["entries"]
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["segment_id"], "seg-1")

        clear_response = self.client.delete("/api/rewrite/audit")
        self.assertEqual(clear_response.status_code, 200)

    def test_admin_tuned_profile_get(self):
        response = self.client.get("/api/admin/tuned-profile")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("tuned_config", payload)

    def test_admin_login_logout_session(self):
        login = self.client.post(
            "/api/admin/login",
            json={"username": os.getenv("ADMIN_USERNAME", "admin"), "password": os.getenv("ADMIN_PASSWORD", "")},
        )
        self.assertEqual(login.status_code, 200)

        session_status = self.client.get("/api/admin/session")
        self.assertEqual(session_status.status_code, 200)
        self.assertTrue(session_status.get_json()["authenticated"])

        logout = self.client.post("/api/admin/logout")
        self.assertEqual(logout.status_code, 200)

        session_status_after = self.client.get("/api/admin/session")
        self.assertEqual(session_status_after.status_code, 200)
        self.assertFalse(session_status_after.get_json()["authenticated"])

    def test_admin_login_invalid_credentials(self):
        response = self.client.post(
            "/api/admin/login",
            json={"username": "wrong", "password": "wrong"},
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_tuned_profile_set_unauthorized(self):
        response = self.client.post("/api/admin/tuned-profile", json={"profile": "bbc"})
        self.assertEqual(response.status_code, 403)

    def test_admin_tuned_profile_set_success(self):
        admin_key = os.getenv("ADMIN_API_KEY", "")
        response = self.client.post(
            "/api/admin/tuned-profile",
            json={"profile": "bbc"},
            headers={"X-Admin-Key": admin_key},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["tuned_config"]["profile"], "bbc")

        restore = self.client.post(
            "/api/admin/tuned-profile",
            json={"profile": "pairwise"},
            headers={"X-Admin-Key": admin_key},
        )
        self.assertEqual(restore.status_code, 200)

    def test_admin_tuned_profile_set_success_with_session(self):
        login = self.client.post(
            "/api/admin/login",
            json={"username": os.getenv("ADMIN_USERNAME", "admin"), "password": os.getenv("ADMIN_PASSWORD", "")},
        )
        self.assertEqual(login.status_code, 200)

        response = self.client.post("/api/admin/tuned-profile", json={"profile": "bbc"})
        self.assertEqual(response.status_code, 200)

        restore = self.client.post("/api/admin/tuned-profile", json={"profile": "pairwise"})
        self.assertEqual(restore.status_code, 200)


if __name__ == "__main__":
    unittest.main()
