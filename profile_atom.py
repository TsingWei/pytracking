import importlib
import torch
import time
from collections import OrderedDict

tracker_name = 'atom'
config_name = 'default'
tracker_module = importlib.import_module('pytracking.tracker.{}'.format(tracker_name))
param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(tracker_name, config_name))
params = param_module.parameters()

tracker_class = tracker_module.get_tracker_class()
tracker = tracker_class(params)

use_gpu = True
torch.cuda.set_device(0)
# model = tracker.params.net

# xs = params.image_sample_size
# zs = params.image_template_size
xs = 256
zs = 128

next_object_id = 1
sequence_object_ids = []
prev_output = OrderedDict()
output_boxes = OrderedDict()
total_frames = 500
dummy_video = (torch.rand(total_frames, xs, xs, 3)*255).numpy()
# x = (torch.rand(xs, xs, 3)*255).numpy()
box = [25.0, 25.0, 1.0, 1.0]
# box = torch.randn(100, 100, 50, 50) # xywh
# z = torch.randn(1, 3, zs, zs)
# zf = torch.randn(1, 96, 8, 8)
# if use_gpu:
#     model = model.cuda()
#     x = x.cuda()
#     box = box.cuda()
    # z = z.cuda()
# zf = model.template(z)

T_w = 10  # warmup
T_t = 100  # test
with torch.no_grad():
    tracker.initialize_features()
    out = tracker.initialize(dummy_video[0], {'init_bbox': box,
                                       'init_object_ids': [next_object_id, ],
                                       'object_ids': [next_object_id, ],
                                       'sequence_object_ids': [next_object_id, ]})
    prev_output = OrderedDict(out)

    output_boxes[next_object_id] = [box, ]
    sequence_object_ids.append(next_object_id)
    next_object_id += 1

    for i in range(T_w):
        info = OrderedDict()
        info['previous_output'] = prev_output
        info['sequence_object_ids'] = sequence_object_ids
        out = tracker.track(dummy_video[0], info)
        prev_output = OrderedDict(out)
    t_s = time.time()
    for i in range(T_t):
        info = OrderedDict()
        info['previous_output'] = prev_output
        info['sequence_object_ids'] = sequence_object_ids
        out = tracker.track(x, info)
        prev_output = OrderedDict(out)
    torch.cuda.synchronize()
    t_e = time.time()
    print('speed: %.2f FPS' % (T_t / (t_e - t_s)))


print("Done")
