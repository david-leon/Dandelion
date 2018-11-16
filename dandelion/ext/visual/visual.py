# coding:utf-8
'''
  Model summary and visualization toolkits
  Created   :  11, 15, 2018
  Revised   :  11, 16, 2018  add `size_unit` arg for `get_model_summary()`
  All rights reserved
'''
__author__ = 'dawei.leng'

from collections import OrderedDict

def get_model_size(model):
    """
    Calculate model parameter size
    :param model:
    :return: total size of model, in bytes
    """
    weights = model.get_weights()
    model_size = 0
    for weight, _ in weights:
        model_size += weight.size * weight.itemsize
    return model_size

def get_model_summary(model, size_unit='M'):
    """
    Produce model parameter summary
    :param model:
    :param size_unit: {'M'|'K'|'B'|int}
    :return: OrderedDict
    """
    if isinstance(size_unit, str):
        size_unit = size_unit.upper()
        assert size_unit in {'M', 'K', 'B'}
        if size_unit == 'M':
            size_unit = 1048576
        elif size_unit == 'K':
            size_unit = 1024
        else:
            size_unit = 1
    model_summary = OrderedDict()
    model_summary['name'] = model.name
    model_size = get_model_size(model) / size_unit
    model_summary['size'] = model_size
    if len(model.params) > 0:
        model_summary['params'] = []
        for param in model.params:
            v = param.get_value()
            model_summary['params'].append(OrderedDict([('name',param.name), ('shape',str(v.shape)), ('dtype',v.dtype.name), ('size',v.size * v.itemsize/size_unit), ('percent',v.size * v.itemsize/(model_size*size_unit)*100)]))
    if len(model.self_updating_variables) > 0:
        model_summary['self_updating_variables'] = []
        for param in model.self_updating_variables:
            v = param.get_value()
            model_summary['self_updating_variables'].append(OrderedDict([('name',param.name), ('shape',str(v.shape)), ('dtype',v.dtype.name), ('size',v.size * v.itemsize/size_unit), ('percent',v.size * v.itemsize/(model_size*size_unit)*100)]))
    if len(model.sub_modules) > 0:
        sub_module_summaries = []
        for tag, child in (model.sub_modules.items()):
            sub_module_summaries.append(get_model_summary(child, size_unit=size_unit))
            sub_module_summaries[-1]['percent'] = sub_module_summaries[-1]['size'] / model_size * 100
        model_summary.update({'sub_modules': sub_module_summaries})
    return model_summary


if __name__ == '__main__':
    import json
    from dandelion.model import Alternate_2D_LSTM
    input_dim, hidden_dim, B, H, W = 8, 8, 2, 32, 32
    model = Alternate_2D_LSTM(input_dims=[input_dim],hidden_dim=hidden_dim, peephole=True, mode=2)
    model_summary = get_model_summary(model, size_unit=1)
    rpt = json.dumps(model_summary, ensure_ascii=False, indent=2)
    print(rpt)

