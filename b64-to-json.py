# uv run b64-to-json.py b64/robot-extension.b64
import base64
import zlib
import sys
import json
import os

def decode_blueprint(blueprint_string):
    decoded = base64.b64decode(blueprint_string[1:])  # Skip the version byte
    json_data = zlib.decompress(decoded).decode('utf-8')

    # print("mutating")
    obj = json.loads(json_data)
    obj['blueprint']['description'] = 'No description yet.'
    for k, v in obj['blueprint'].items():
        if type(v) is list:
            print(f'    [{k}]: saving {len(obj["blueprint"]["entities"])} entities')
        else:
            print(f'    [{k}]: saving `{v}`')
    json_data = json.dumps(obj)

    # print("writing")
    path = 'blueprints/' + blueprint_string[:20].replace('/', '_') + '.json'
    with open(path, 'w') as f:
        f.write(json_data)
    print('    written as ' + path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} blueprint_string [blueprint_string]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        print(f"decoding {arg[:30]}...")
        decode_blueprint(arg)


