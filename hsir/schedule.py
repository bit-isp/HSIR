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

denoise_restormer = TrainSchedule(
    max_epochs=80,
    base_lr=1e-4,
    lr_schedule={
        0: 1e-4,
        45: 5e-5,
        55: 1e-5,
        65: 5e-6,
        75: 1e-6,
    },
)

denoise_hsid_cnn = denoise_restormer

denoise_complex_default = TrainSchedule(
    max_epochs=110,
    base_lr=1e-3,
    lr_schedule={
        80: 1e-3,
        90: 5e-4,
        95: 1e-4,
        100: 5e-5,
        105: 1e-5,
    },
)

denoise_complex_restormer = TrainSchedule(
    max_epochs=110,
    base_lr=1e-4,
    lr_schedule={
        80: 1e-4,
        82: 5e-5,
        90: 1e-5,
        95: 5e-6,
        100: 1e-6
    },
)

denoise_complex_hsid_cnn = TrainSchedule(
    max_epochs=110,
    base_lr=1e-4,
    lr_schedule={
        80: 1e-4,
        95: 5e-5,
        100: 1e-5,
        105: 1e-6,
    },
)
