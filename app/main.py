import io
import json
import os
import tempfile
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from multipart import parse_form

from app.clearer_engine import remove_background

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD = 15 * 1024 * 1024
OUTPUT_DIR = Path("output")
MODEL_PATH = Path("app/models/u2netp.pth")
ENGINE_LOCK = threading.Lock()


def parse_multipart(content_type, body):
    fields = {}

    def decode(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value)

    def on_field(field):
        name = decode(field.field_name)
        fields[name] = {
            "filename": None,
            "data": field.value
        }

    def on_file(upload):
        name = decode(upload.field_name)
        filename = decode(upload.file_name)
        upload.file_object.seek(0)
        data = upload.file_object.read()
        fields[name] = {
            "filename": filename,
            "data": data
        }

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body))
    }

    parse_form(headers, io.BytesIO(body), on_field, on_file)
    return fields


class ClearerHandler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            page = Path(__file__).resolve().parent / "static" / "index.html"
            try:
                content = page.read_bytes()
            except OSError:
                return self.send_json(500, {
                    "error": "Web interface not found",
                    "detail": str(page)
                })

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/health":
            return self.send_json(200, {
                "status": "ok",
                "model_file_exists": MODEL_PATH.is_file()
            })

        return self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/remove-background":
            return self.send_json(404, {"error": "Not found"})

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type.lower():
            return self.send_json(400, {
                "error": "Use multipart/form-data with a file field"
            })

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_json(400, {"error": "Invalid content length"})

        if length <= 0 or length > MAX_UPLOAD:
            return self.send_json(413, {
                "error": "Upload must be between 1 byte and 15 MB"
            })

        body = self.rfile.read(length)

        try:
            fields = parse_multipart(content_type, body)
        except Exception as exc:
            print("Multipart parsing error:", repr(exc), flush=True)
            return self.send_json(400, {
                "error": "Invalid multipart request",
                "detail": str(exc)
            })

        uploaded = fields.get("file")
        if not uploaded or not uploaded["filename"] or not uploaded["data"]:
            return self.send_json(400, {
                "error": "Missing image upload field named file"
            })

        try:
            size_value = fields.get("size", {}).get("data", b"384")
            if isinstance(size_value, bytes):
                size_value = size_value.decode("ascii", "ignore")
            inference_size = int(size_value)
        except (TypeError, ValueError):
            return self.send_json(400, {"error": "Invalid inference size"})

        if inference_size not in (320, 384, 512):
            return self.send_json(400, {
                "error": "Size must be 320, 384 or 512"
            })

        input_path = None
        output_path = None

        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                prefix="clearer_input_",
                suffix=".upload",
                dir=str(OUTPUT_DIR),
                delete=False
            ) as temp_input:
                input_path = Path(temp_input.name)
                temp_input.write(uploaded["data"])

            fd, output_name = tempfile.mkstemp(
                prefix="clearer_result_",
                suffix=".png",
                dir=str(OUTPUT_DIR)
            )
            os.close(fd)
            output_path = Path(output_name)

            with ENGINE_LOCK:
                remove_background(
                    str(input_path),
                    str(output_path),
                    inference_size=inference_size
                )

            png = output_path.read_bytes()

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header(
                "Content-Disposition",
                'attachment; filename="clearer_result.png"'
            )
            self.end_headers()
            self.wfile.write(png)

        except Exception as exc:
            print("Clearer.bc API error:", repr(exc), flush=True)
            try:
                self.send_json(500, {
                    "error": "Image processing failed",
                    "detail": str(exc)
                })
            except (BrokenPipeError, ConnectionResetError):
                pass

        finally:
            for path in (input_path, output_path):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def log_message(self, fmt, *args):
        print("[Clearer.bc]", fmt % args, flush=True)


if __name__ == "__main__":
    print("Clearer.bc local API", flush=True)
    print(f"Listening at http://{HOST}:{PORT}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    server = ThreadingHTTPServer((HOST, PORT), ClearerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Clearer.bc API...", flush=True)
    finally:
        server.server_close()
