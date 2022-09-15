import re
from PIL import Image
from io import BytesIO
import base64

def copy_res_to_img(img, index):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img[index - 1])
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        return processed_image
    except IndexError:
        print("index exceeds range...")
        return None