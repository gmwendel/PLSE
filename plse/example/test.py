from plse.tools.plots import make_plots
from plse.traincounter import train_counter

#train_counter(["test_10pct_3MeV.npz", ], "my_plse_counter", save_history=False, use_multiprocessing=False)

input_files = ["test_1pct_3MeV.npz", ]
input_network = "plse_counter"
output_plots = "test_plots"

make_plots(input_files, input_network, output_plots)
