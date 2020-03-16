import time

class CvTimer:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total_time = 0
        pass

    def __enter__(self):
        self.count += 1
        self.start_time = time.time()
        pass

    def __exit__(self, type, value, tb):
        elapse = time.time() - self.start_time
        self.total_time += elapse

    def __repr__(self):
        res = 'Timer: %-8s    Count: %-4d    Time: %-8.2f    FPS: %-8.2f' % (self.name, self.count, self.total_time / self.count, self.count / self.total_time)
        return res


if __name__ == '__main__':
    from tqdm import tqdm
    sleep1_timer = CvTimer('1s')
    for i in tqdm(range(20)):
        with sleep1_timer:
            print('Hello world.')
            time.sleep(0.2)

    print(sleep1_timer)
