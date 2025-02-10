# uv run json-to-b64.py myfile.json
import base64
import zlib
import sys
import json

def encode_blueprint(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    compressed = zlib.compress(json.dumps(json_data).encode('utf-8'))
    b64_encoded = base64.b64encode(compressed).decode('utf-8')

    blueprint_string = '0' + b64_encoded  # Add version byte
    print(blueprint_string)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} json_file")
        sys.exit(1)

    encode_blueprint(sys.argv[1])
