import io
from functools import lru_cache
from pathlib import Path

import dvc.api
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.http import InferenceServerClient, InferRequestedOutput


def load_test(test_name: str):
    assert test_name in ["0", "8"], "test doesn't exist. try 0 or 8"

    image_name = test_name + ".png"
    image_info = dvc.api.read(
        str(Path("./triton_backend_cpu") / "tests" / image_name),
        repo="https://github.com/rw404/mnist-sandbox",
        mode="rb",
    )
    image = Image.open(io.BytesIO(image_info))

    return image


def preprocess(image):
    image = image.resize((28, 28))
    image = np.array(image).astype(np.float32)
    image = image.reshape(1, 1, 28, 28)

    return image


def postprocess_output(preds):
    return np.argmax(np.squeeze(preds))


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_image_onnx(image: np.ndarray):
    triton_client = get_client()

    inputs = []
    inputs.append(httpclient.InferInput("input.1", [1, 1, 28, 28], "FP32"))
    inputs[0].set_data_from_numpy(image)

    infer_output = InferRequestedOutput("26")

    query_response = triton_client.infer("onnx-mnist-cpu", inputs, outputs=[infer_output])
    prediction = query_response.as_numpy("26")[0]

    class_img = postprocess_output(prediction)
    return class_img


def test_image(class_img):
    test_img = load_test(class_img)
    normalized_image = preprocess(test_img)
    prediction = call_triton_image_onnx(normalized_image)

    assert str(prediction) == class_img, f"Error with image {prediction}"
    print(f"Test for image of {class_img} class is done.")


def main():
    # Test for 0 class
    test_image("0")

    # Test for 8 class
    test_image("8")


if __name__ == "__main__":
    main()
