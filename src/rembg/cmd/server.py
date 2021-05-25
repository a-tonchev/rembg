import os
import glob
import argparse
from io import BytesIO
from urllib.parse import unquote_plus
from urllib.request import urlopen
import base64
from flask import Flask, request, send_file, jsonify
from waitress import serve

from ..bg import remove
from PIL import Image

app = Flask(__name__)

def img_to_base64_str(buf):
    img = Image.open(buf)
    img.save(buf, format="PNG")
    buf.seek(0)
    img_byte = buf.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str

@app.route("/", methods=["GET", "POST"])
def index():
    file_content = ""

    if request.method == "POST" and "image_data" in request.json:
        json_data = request.get_json()
        image_data = json_data['image_data']
        file_content = base64.b64decode(image_data)
        alpha_matting = json_data['a'] or False
        af = json_data['af'] or 240
        ab = json_data['ab'] or 10
        ae = json_data['ae'] or 10
        az = json_data['az'] or 1000

    if request.method == "GET":
        url = request.args.get("url", type=str)
        if url is None:
            return {"error": "missing query param 'url'"}, 400

        file_content = urlopen(unquote_plus(url)).read()
        alpha_matting = "a" in request.values
        af = request.values.get("af", type=int, default=240)
        ab = request.values.get("ab", type=int, default=10)
        ae = request.values.get("ae", type=int, default=10)
        az = request.values.get("az", type=int, default=1000)

    if file_content == "":
        return {"error": "File content is empty"}, 400

    model = request.args.get("model", type=str, default="u2net")
    model_path = os.environ.get(
        "U2NETP_PATH",
        os.path.expanduser(os.path.join("~", ".u2net")),
    )
    model_choices = [os.path.splitext(os.path.basename(x))[0] for x in set(glob.glob(model_path + "/*"))]
    if len(model_choices) == 0:
        model_choices = ["u2net", "u2netp", "u2net_human_seg"]

    if model not in model_choices:
        return {"error": f"invalid query param 'model'. Available options are {model_choices}"}, 400

    try:
        file_without_background = remove(
                                         file_content,
                                         model_name=model,
                                         alpha_matting=alpha_matting,
                                         alpha_matting_foreground_threshold=af,
                                         alpha_matting_background_threshold=ab,
                                         alpha_matting_erode_structure_size=ae,
                                         alpha_matting_base_size=az,
                                     )
        file_data = BytesIO(file_without_background)
        encoded = img_to_base64_str(file_data)
        return jsonify(
            converted_image=encoded,
        )
    except Exception as e:
        app.logger.exception(e, exc_info=True)
        return {"error": "oops, something went wrong!"}, 500


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-a",
        "--addr",
        default="0.0.0.0",
        type=str,
        help="The IP address to bind to.",
    )

    ap.add_argument(
        "-p",
        "--port",
        default=9339,
        type=int,
        help="The port to bind to.",
    )

    args = ap.parse_args()
    serve(app, host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
