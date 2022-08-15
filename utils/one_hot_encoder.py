import torch


def one_hot_encode(label, label_values=torch.arange(183)):
    semantic_map = []
    for colour in label_values:
        equality = torch.eq(label, colour)
        semantic_map.append(equality)
    semantic_map = torch.stack(semantic_map, dim=-1)

    return semantic_map


def reverse_one_hot(image):
    x = torch.argmax(image.int(), dim=-1)
    return x
