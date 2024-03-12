from math import sqrt


class Config:
    SEED = 42
    CANVAS_SIZE = (952, 1360)
    EARLY_STOPPING_EPOCH = 7
    PRINT_FREQ = 66  # 54

    @staticmethod
    def divisor_generator(n):
        large_divisors = []
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                yield i
                if i * i != n:
                    large_divisors.append(n / i)
        for divisor in reversed(large_divisors):
            yield divisor
