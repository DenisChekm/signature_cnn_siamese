from torch.nn import Conv2d, BatchNorm2d, Linear, BatchNorm1d, MaxPool2d


def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(flatten_model(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(flatten_model(n))
        except:
            ret.append(modules)
    return flatten_list(ret)


def prin_layers_info(model):
    print("======================================================================")
    print("Model structure")
    print("======================================================================")
    target_layers = []
    module_list = [module for module in model.modules()]  # this is needed
    flatted_list = flatten_model(module_list)

    for _, value in enumerate(flatted_list):

        if isinstance(value, (Conv2d, MaxPool2d, BatchNorm2d, Linear, BatchNorm1d)):
            print(value)
            target_layers.append(value)

    print("======================================================================")
    print("======================================================================")
