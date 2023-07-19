
def postkdearg():

    import argparse

    parser = argparse.ArgumentParser(description='PINN of convection-dominated 1D flows on Lagrangian framework')

    # Case parameters
    parser.add_argument('--CASENO', type=int, default=1, choices=range(1, 5), \
                        help='Case number')

    parser.add_argument('--NLES', type=int, default=32, choices=range(1, 2048), \
                        help='Grid resoltuion')

    parser.add_argument('--METHOD', type=str, default='Dsmag',\
                        choices=['Smag', 'Leith', 'DLeith', 'DSmag', 'RLLeith','RLSmag'], \
                        help='Method for SGS')

    parser.add_argument('--SPIN_UP', type=int, default=50_000, \
                        help='Spin up ')

    parser.add_argument('--NUM_DATA', type=int, default=2_000, \
                        help='Maximum number of data points')

    parser.add_argument('--NumRLSteps', type=float, default=1_000.0, \
                        help='Maximum number of data points')

    parser.add_argument('--EPERU', type=float, default=1.0, \
                        help='Episodes per update')

    parser.add_argument('--nAgents', type=int, default=16, \
                        help='Number of agents')

    return parser.parse_args()
