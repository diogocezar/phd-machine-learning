import time


def get_time():
    return time.time()


def get_time_diff(start_time):
    end_time = time.time()
    return round_float(end_time - start_time)


def round_float(value):
    return float("{:.3f}".format(value))
