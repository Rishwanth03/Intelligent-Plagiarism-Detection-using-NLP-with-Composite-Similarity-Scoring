from flask import Flask
from flask_cors import CORS

from app.routes.admin import admin_bp
from app.routes.detection import detection_bp
from app.routes.documents import documents_bp
from app.routes.reports import reports_bp
from app.routes.rewrite import rewrite_bp
from app.routes.similarity import similarity_bp
from app.services.container import rewrite_service, similarity_service


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object("app.config.Config")
    CORS(app, supports_credentials=True)

    app.register_blueprint(documents_bp, url_prefix="/api/documents")
    app.register_blueprint(similarity_bp, url_prefix="/api/similarity")
    app.register_blueprint(detection_bp, url_prefix="/api/detect")
    app.register_blueprint(reports_bp, url_prefix="/api/report")
    app.register_blueprint(rewrite_bp, url_prefix="/api/rewrite")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")

    @app.route("/", methods=["GET"])
    def index():
        return {
            "service": "Automated Plagiarism Detection Backend",
            "status": "running",
            "health": "/api/health",
        }, 200

    @app.route("/api", methods=["GET"])
    def api_index():
        return {
            "routes": [
                "/api/health",
                "/api/documents",
                "/api/similarity/composite",
                "/api/similarity/full-analysis",
                "/api/detect",
                "/api/report/statistics",
                "/api/rewrite/suggest",
                "/api/admin/tuned-profile",
            ]
        }, 200

    @app.route("/api/health", methods=["GET"])
    def healthcheck():
        return {
            "status": "ok",
            "services": {
                "similarity": {
                    "trained": similarity_service.detector.is_trained,
                    "trained_corpus_size": similarity_service.detector.trained_corpus_size,
                    "tuned_config": similarity_service.detector.get_tuned_config(),
                },
                "rewrite": rewrite_service.status(),
            },
        }, 200

    return app
