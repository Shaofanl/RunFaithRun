import os
import numpy as np
from scipy.misc import imread

if __name__ == '__main__':
    # start_time = 1520288889.2129278
    # start_time = 1520289193.6855888
    # start_time = 1520323553.994526
    # start_time = 1520369935.8181005
    start_time = 1520411037.6885858
    filedir = os.path.abspath("./data/"+str(start_time))

    from RunFaithRun.utils.data_utils import load_image_from_dir
    images, images_time = load_image_from_dir(filedir)
    images_time -= start_time

    # load log
    log = []
    log_time = []
    with open(os.path.join(filedir, 'log.txt')) as f:
        for line in f:
            line = line.strip().split(':')
            log.append(line[2])
            log_time.append(float(line[3][1:-1])-start_time)
    logs = np.array(log)
    logs_time = np.array(log_time)

    # make dataset
    def generate_data(
            images, images_time,
            logs, logs_time, log_filter,
            window_size=3):
        # filter
        _logs, _logs_time = [], []
        for log, log_time in zip(logs, logs_time):
            if log_filter(log):
                _logs.append(log)
                _logs_time.append(log_time)

        # label all timetags
        all_times = sorted(_logs_time+list(images_time))
        all_labels = [None] * len(all_times)
        for ind, t in enumerate(all_times):
            if t in _logs_time:
                all_labels[ind] = \
                    1 if _logs[_logs_time.index(t)].split()[1] == 'pressed' else 0
            # if all_labels[ind] == 1:
            #     all_labels[ind-1] = 1  # extend the space press forward

        # fill the None in all labels
        for ind, ele in enumerate(all_labels):
            if ele is None:
                if ind == 0:
                    all_labels[ind] = 0
                else:
                    all_labels[ind] = all_labels[ind-1]
        print (all_labels)

        # create dataset
        x, y = [], []
        xi = []
        for img_ind, time in zip(range(len(images)), images_time):
            xi.append(img_ind)
            label = all_labels[all_times.index(time)]
            if len(xi) >= window_size:
                x.append(np.array(xi))
                y.append(label)
                xi.pop(0)
        x, y = np.array(x), np.array(y)
        return x, y

    # space_x: indices
    space_x, space_y = generate_data(
        images, images_time,
        logs, logs_time,
        log_filter=lambda x: x.startswith('space'),
        window_size=3
    )
    print (space_x.shape)
    print (space_y.shape)

    # visualize
    # import cv2
    # for x, y in zip(space_x, space_y):
    #     print (y)
    #     frame = np.concatenate(x, 1)
    #     print (frame.shape)
    #     frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(0) == 27:
    #         break
    # cv2.destroyAllWindows()

    # space_x = space_x.swapaxes(1, 3).swapaxes(1, 2)
    # space_x = space_x.reshape(space_x.shape[:-2] + (9,))
    print(space_x.shape)
    np.savez_compressed('space_data.npz',
                        x=space_x, y=space_y, filedir=filedir)
