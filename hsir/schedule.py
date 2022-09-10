from collections import namedtuple


TrainSchedule = namedtuple('TrainSchedule', ['base_lr', 'lr_schedule', 'max_epochs'])


denoise_default = TrainSchedule(
    max_epochs=80,
    base_lr=1e-3,
    lr_schedule={
        0: 1e-3,
        20: 1e-4,
        30: 1e-3,
        45: 1e-4,
        55: 5e-5,
        60: 1e-5,
        65: 5e-6,
        75: 1e-6,
    },
)
