"""
Burner script to explore the Logistic and TabNet model with multple
training runs to explore if they are in fact different at all.


"""

# log: the list variablbes I copy and paste with the logistic data
# tab: the list variablbes I copy and paste with the tabnet data
log = []; tab = [] # Place older

from src.utils import plot_execution_times_histogram

plot_execution_times_histogram(log)

plot_execution_times_histogram(tab)