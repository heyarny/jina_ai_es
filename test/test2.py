from docarray import DocumentArray, Document

#instead of manually specifying embedding, use a DNN with .embed option
q = (Document(uri="../images/pear.png")
     .load_uri_to_image_tensor()
     .set_image_tensor_normalization()
     .set_image_tensor_channel_axis(-1, 0))

#embed it into a vector
import torchvision
model = torchvision.models.resnet50(pretrained=True)
d = q.embed(model)

