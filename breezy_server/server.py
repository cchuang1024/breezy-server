import uuid
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from pydantic import ValidationError
from celery.result import AsyncResult

from .base_models import (
    NormalResponse,
    ErrorResponse,
    TransformResponse,
    TransformRequest,
    TaskStatusResponse,
)
from .async_inference import inference_task

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "sample"
TARGET_DIR = BASE_DIR / "target"

UPLOAD_DIR.mkdir(exist_ok=True)
TARGET_DIR.mkdir(exist_ok=True)


@app.route("/")
def home():
    """回傳首頁 HTML"""
    return send_file(BASE_DIR / "template" / "index.html")


@app.route("/api/v1/sample/upload", methods=["POST"])
def upload_sample():
    """上傳 .wav 檔案"""
    if "file" not in request.files:
        return (
            jsonify(ErrorResponse(error="No file part in the request").model_dump()),
            400,
        )

    file = request.files["file"]

    if file.filename == "":
        return jsonify(ErrorResponse(error="No selected file").model_dump()), 400

    if file and file.filename.endswith(".wav"):
        filename = "sample.wav"
        file_path = UPLOAD_DIR / filename

        try:
            file.save(file_path)
            return (
                jsonify(
                    NormalResponse(message="File uploaded successfully").model_dump()
                ),
                200,
            )
        except Exception as e:
            return (
                jsonify(
                    ErrorResponse(error=f"File upload failed: {str(e)}").model_dump()
                ),
                500,
            )

    return (
        jsonify(
            ErrorResponse(
                error="Invalid file type. Only .wav files are accepted"
            ).model_dump()
        ),
        400,
    )


@app.route("/api/v1/sample/create", methods=["POST"])
def create_sample():
    """建立 sample.txt"""
    data = request.get_json(silent=True)

    if not data and "sample" not in data:
        return jsonify(ErrorResponse(error="Invalid request data").model_dump()), 400

    sample_text = data["sample"]
    file_path = UPLOAD_DIR / "sample.txt"

    try:
        with open("./sample/sample.txt", "w") as f:
            f.write(sample_text)
        return (
            jsonify(
                NormalResponse(message="Sample text saved successfully").model_dump()
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                ErrorResponse(
                    error=f"Failed to save sample text: {str(e)}"
                ).model_dump()
            ),
            500,
        )


@app.route("/api/v1/transform", methods=["POST"])
def transform():
    """將資料存入 target 資料夾"""
    try:
        data = TransformRequest(**request.json)
    except ValidationError as e:
        return jsonify(ErrorResponse(error=str(e)).model_dump()), 400

    task = inference_task.apply_async(args=[data.target])

    return jsonify(TransformResponse(task_id=task.id).model_dump()), 202

    # data = request.get_json()

    # if not data or "target" not in data:
    #     return jsonify(NormalResponse(message="Invalid request data")), 400

    # target_text = data["target"]
    # unique_id = uuid.uuid4()
    # directory = TARGET_DIR / str(unique_id)

    # try:
    #     directory.mkdir(exist_ok=True)
    #     file_path = directory / "target.txt"

    #     with file_path.open("w", encoding="utf-8") as f:
    #         f.write(target_text)

    #     return jsonify(TransformResponse(receipt_id=str(unique_id))), 200
    # except Exception as e:
    #     return jsonify(ErrorResponse(error=f"Transformation failed: {str(e)}")), 500


@app.route("/api/v1/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """取得 task 狀態"""
    task = inference_task.AsyncResult(task_id)

    if task.state == "PENDING":
        return (
            jsonify(TaskStatusResponse(task_id=task_id, status="pending").model_dump()),
            202,
        )
    elif task.state == "SUCCESS":
        return (
            jsonify(
                TaskStatusResponse(
                    task_id=task_id,
                    status="completed",
                    receipt_id=task.result["receipt_id"],
                ).model_dump()
            ),
            200,
        )
    elif task.state == "FAILURE":
        return (
            jsonify(
                TaskStatusResponse(
                    task_id=task_id, status="failed", error=str(task.result["error"])
                ).model_dump()
            ),
            500,
        )
    else:
        return (
            jsonify(
                TaskStatusResponse(task_id=task_id, status=task.state).model_dump()
            ),
            202,
        )


@app.route("/api/v1/task/<task_id>/result", methods=["GET"])
def get_transformed_result(task_id):
    """下載轉換後的 WAV 檔案"""
    task_result = AsyncResult(task_id)

    if task_result.state == "PENDING":
        return (
            jsonify(TaskStatusResponse(task_id=task_id, status="pending").model_dump()),
            202,
        )
    elif task_result.state == "FAILURE":
        return jsonify(
            TaskStatusResponse(
                task_id=task_id, status="failed", error=str(task_result.result["error"])
            ).model_dump(),
            500,
        )
    elif task_result.state == "SUCCESS":
        receipt_id = task_result.result["receipt_id"]
        result_path = TARGET_DIR / receipt_id / "result.wav"

        if result_path.exists():
            return send_file(result_path, as_attachment=True)
        else:
            return (
                jsonify(
                    TaskStatusResponse(
                        task_id=task_id,
                        status="completed",
                        error="Result file not found",
                    ).model_dump()
                ),
                500,
            )
    else:
        return (
            jsonify(
                TaskStatusResponse(
                    task_id=task_id, status=task_result.state
                ).model_dump()
            ),
            202,
        )


if __name__ == "__main__":
    app.run(debug=True)
