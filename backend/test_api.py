"""
Automated PyTest Test Suite for FastAPI Crop Disease Endpoints
Author: Khushi Yadav
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["supported_classes"] == 38
    assert data["author"] == "Khushi Yadav"


def test_classes_endpoint():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 38
    assert "Tomato___Late_blight" in data["classes"]
    assert "Potato___Early_blight" in data["classes"]


def test_treatment_endpoint_hindi():
    response = client.get("/treatment/Tomato___Late_blight?lang=hi")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "टमाटर"
    assert "झुलसा" in data["name"]
    assert "सल्फर" in data["organic_treatment"] or "नीम" in data["organic_treatment"]
    assert "मैनकोज़ेब" in data["chemical_treatment"] or "रिडोमिल" in data["chemical_treatment"]


def test_treatment_endpoint_english():
    response = client.get("/treatment/Potato___Early_blight?lang=en")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Potato"
    assert "Early Blight" in data["name"]
    assert "Alternaria" in data["scientific_name"]


def test_predict_endpoint():
    # Create synthetic test leaf image in memory
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict?lang=hi",
        files={"file": ("test_leaf.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_3" in data
    assert len(data["top_3"]) == 3
    assert "agronomy" in data
