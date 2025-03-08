import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from flask import jsonify

from breezy_server.server import app
from breezy_server.base_models import (
    TaskStatusResponse,
    TransformRequest,
    TransformResponse,
    NormalResponse,
    ErrorResponse,
    SampleCreateRequest,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "sample"
TARGET_DIR = BASE_DIR / "target"


@pytest.fixture
def client():
    """設定 Flask 測試 Client"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    """測試首頁應該回傳 200"""
    response = client.get("/")
    assert response.status_code == 200


def test_upload_sample_success(client):
    """測試成功上傳 .wav 檔案"""
    data = {"file": (io.BytesIO(b"fake wav content"), DATA_DIR / "example.wav")}

    response = client.post(
        "/api/v1/sample/upload", data=data, content_type="multipart/form-data"
    )

    print(response)

    assert response.status_code == 200
    assert response.json["message"] == "File uploaded successfully"


def test_upload_sample_no_file(client):
    """測試上傳時缺少檔案"""
    response = client.post(
        "/api/v1/sample/upload", data={}, content_type="multipart/form-data"
    )
    assert response.status_code == 400
    assert "error" in response.json


def test_upload_sample_wrong_format(client):
    """測試上傳非 .wav 檔案"""
    data = {"file": (io.BytesIO(b"fake content"), DATA_DIR / "test.txt")}
    response = client.post(
        "/api/v1/sample/upload", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 400
    assert "error" in response.json


def test_create_sample_success(client):
    """測試成功建立 sample.txt"""
    response = client.post(
        "/api/v1/sample/create",
        json={
            "sample": "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。"
        },
    )

    assert response.status_code == 200
    assert response.json["message"] == "Sample text saved successfully"


def test_create_sample_missing_field(client):
    """測試缺少 sample 欄位"""
    response = client.post("/api/v1/sample/create", json={})
    assert response.status_code == 400
    assert "error" in response.json


@patch("breezy_server.async_inference.inference_task.apply_async")
def test_transform_success(mock_celery, client):
    """測試成功提交 transform"""
    mock_celery.return_value.id = "mock_task_id"
    response = client.post("/api/v1/transform", json={"target": "Hello"})
    assert response.status_code == 202
    assert response.json["task_id"] == "mock_task_id"


def test_transform_invalid_request(client):
    """測試錯誤的 transform JSON"""
    response = client.post("/api/v1/transform", json={})
    assert response.status_code == 400
    assert "error" in response.json


@patch("breezy_server.async_inference.inference_task.AsyncResult")
def test_get_task_status_pending(mock_result, client):
    """測試查詢 PENDING 任務"""
    mock_result.return_value.state = "PENDING"
    response = client.get("/api/v1/task/mock_task_id")
    assert response.status_code == 202
    assert response.json["status"] == "pending"


@patch("breezy_server.async_inference.inference_task.AsyncResult")
def test_get_task_status_success(mock_result, client):
    """測試查詢已完成任務"""
    mock_result.return_value.state = "SUCCESS"
    mock_result.return_value.result = {"receipt_id": "1234"}
    response = client.get("/api/v1/task/mock_task_id")
    assert response.status_code == 200
    assert response.json["status"] == "completed"
    assert response.json["receipt_id"] == "1234"


@patch("breezy_server.async_inference.inference_task.AsyncResult")
def test_get_task_status_failure(mock_result, client):
    """測試查詢失敗的任務"""
    mock_result.return_value.state = "FAILURE"
    mock_result.return_value.result = {"error": "Something went wrong"}
    response = client.get("/api/v1/task/mock_task_id")
    assert response.status_code == 500
    assert response.json["status"] == "failed"


@patch("breezy_server.async_inference.inference_task.AsyncResult")
def test_get_transformed_result_pending(mock_result, client):
    """測試下載結果時 PENDING"""
    mock_result.return_value.state = "PENDING"
    response = client.get("/api/v1/task/mock_task_id/result")
    assert response.status_code == 202
    assert response.json["status"] == "pending"


@patch("breezy_server.async_inference.inference_task.AsyncResult")
def test_get_transformed_result_success(mock_result, client):
    """測試成功下載 .wav 檔案"""
    mock_result.return_value.state = "SUCCESS"
    mock_result.return_value.result = {"receipt_id": "1234"}

    # 模擬存在的 .wav 檔案
    test_wav_path = TARGET_DIR / "1234" / "result.wav"
    with open(test_wav_path, "wb") as f:
        f.write(b"fake wav content")

    response = client.get("/api/v1/task/mock_task_id/result")
    assert response.status_code == 200 or response.status_code == 202
