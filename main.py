import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from src.database.base import init_db

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './data/raw_documents')
    app.config['VECTOR_DB_DIR'] = os.getenv('VECTOR_DB_DIR', './data/vector_db')
    app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', './models/tf_classifier.h5')
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['VECTOR_DB_DIR'], exist_ok=True)

    from routes.document_routes import document_bp
    from routes.search_routes import search_bp
    from routes.analysis_routes import analysis_bp
    from routes.analytics_routes import analytics_bp

    app.register_blueprint(document_bp, url_prefix='/api/documents')
    app.register_blueprint(search_bp, url_prefix='/api/search')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(analytics_bp, url_prefix='/api/analytics')

    @app.route('/health', methods=['GET'])
    def health_check():
        return {"status": "healthy", "service": "AI Research Assistant API"}, 200

    return app

app = create_app()

if __name__ == '__main__':
    try:
        init_db()
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Failed to start server due to error: {e}")
        import traceback
        traceback.print_exc()