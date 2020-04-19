import logging

from proxsim import Context, master_scheduling
from proxsim.scheduler import MasterSchedulerDummy


def master_callback(simulation_clock):
    print(f'''
        DAG execution time {simulation_clock}
    ''')
    return 'test callback result'

if __name__ == '__main__':
    _LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _LOG_FORMAT = "[%(process)d] [%(name)s] [%(levelname)5s] [%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.WARN, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)


    context_50 = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='./data/workflows/dot/SIPHT.n.50.0.dot',
        master_callback=master_callback
    )

    context_500 = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='./data/workflows/dot/SIPHT.n.500.0.dot',
    )

    for context in [context_50, context_500]:
        result = master_scheduling(context, MasterSchedulerDummy)
        print(f'scheduling result: {result}')
        