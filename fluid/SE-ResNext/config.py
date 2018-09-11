batch_size = -1
image_shape = [3, 224, 224]
class_dim = 102

input_descs = {
    "image": [[batch_size] + image_shape, 'float32'],
    "label": [(batch_size, 1), "int64"]
}

input_fields = (
    "image",
    "label", )
