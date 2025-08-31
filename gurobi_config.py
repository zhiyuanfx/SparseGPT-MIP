MIPGAP = 0.01       # relative MIP optimality gap tolerance. 
BARCONVTOL = 1e-4   # convergence tolerance for the barrier algorithm.
CROSSOVER = 0       # crossover behavior after barrier solve
METHOD = -1         # method for continuous relaxations: -1 = automatic choice 
TIME_LIMIT = 600     # time limit in seconds for the optimization process

params = {
    # fill in your own credentials for Gurobi Web License Service (WLS)
    "WLSACCESSID": "",
    "WLSSECRET":   "",
    "LICENSEID":   1,
    
    "LogToConsole": 0,  # suppress console log
}