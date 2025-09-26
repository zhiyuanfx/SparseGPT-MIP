EPSILON_PERCENT = 1    # |compensation| < epsilon_percent * |original_weight|

# -----Gurobi parameters-----
MIPGAP = 0.01       # relative MIP optimality gap tolerance. 
BARCONVTOL = 1e-4   # convergence tolerance for the barrier algorithm.
CROSSOVER = 0       # crossover behavior after barrier solve
METHOD = -1         # method for continuous relaxations: -1 = automatic choice 
TIME_LIMIT = 6000   # time limit in seconds for the optimization process

USEWLS = True           # whether to use Gurobi Web License Service (WLS)
GUROBI_NUM_PROCESS = 2  # number of processes to use for Gurobi

params = {
    # fill in your own credentials for WLS if USEWLS = True
    "WLSACCESSID": "",
    "WLSSECRET":   "",
    "LICENSEID":   1,
    "LogToConsole": 0  
}

# -----QHD parameters-----
QHD_SHOTS = 100
QHD_STEPS = 10000
QHD_SEED = 42

JAXADAM_ITERS = 300
JAXADAM_LR = 0.1

ADAMMS_ITERS = 300
ADAMMS_LR = 0.1

FORCE_VALID_STANDARD = 'magnitude'  # 'magnitude' or 'sparsegpt'

