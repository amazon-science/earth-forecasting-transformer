import warnings
import os


def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_parameter_names(model, forbidden_layer_types):
    r"""
    Returns the names of the model parameters that are not inside a forbidden layer.

    Borrowed from https://github.com/huggingface/transformers/blob/623b4f7c63f60cce917677ee704d6c93ee960b4b/src/transformers/trainer_pt_utils.py#L996
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
