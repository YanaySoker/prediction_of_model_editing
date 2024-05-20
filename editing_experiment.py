from general_functions import *
from helper_functions import *
from neighborhood import d
from experiment_config import *
import sys

start_line, end_line = int(sys.argv[1]), int(sys.argv[2])
print("lines:", start_line, "to", end_line, "; experiment type: high res")
results = all_results(d[start_line:end_line], HIGH_RES_FLAG = True)
file_name=f"scores_{start_line}_{end_line}.py"
print_list(results, file_name)
