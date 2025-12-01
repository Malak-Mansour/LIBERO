from libero.libero.benchmark import Benchmark


class LiberoCustom(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_custom"
        self._make_benchmark()
