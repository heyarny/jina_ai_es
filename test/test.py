import platform
from docarray import Document

print(platform.machine())


q3 = (Document(uri="/workspace/images/red-apple.jpeg")
     .load_uri_to_image_tensor()
     .set_image_tensor_normalization()
     .set_image_tensor_channel_axis(-1, 0))
