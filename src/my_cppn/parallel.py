from multiprocessing import Pool


# TODO 进程分配方式需要管理，分配进程用于调度和监控
class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):

        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    def evaluate(self, genomes, config):
        jobs = []
        for genome in genomes:
            # 只有一个参数需要写成(xx,)格式
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, genome in zip(jobs, genomes):
            fitness, behavior = job.get(timeout=self.timeout)
            genome.fitness = fitness
            genome.behavior = behavior
