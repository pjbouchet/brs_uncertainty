# https://daft.readthedocs.io/en/latest/

import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=False)

# Instantiate the PGM

pgm = daft.PGM()

# Node placement

node_y = 1
node_x = 1

# Arrow and plate parameters

grey_line = {'linestyle':'solid', 'head_width':0.25, 'edgecolor':'grey', 'facecolor':'lightgrey'}
solid_line = {'linestyle':'solid', 'head_width':0.25}
text_label_noborder = {'edgecolor':'None'}

# Observation model

pgm.add_node("delta", r"$\delta^2$", node_x+1, node_y, shape="rectangle", aspect=1.5)
pgm.add_node("y", r"$y_{i}$", node_x, node_y)

# Process model

 # Expected threshold for whale i
pgm.add_node("mu_i", r"$\mu_{i}$", node_x, node_y-1)

# Mean threshold for all whales
pgm.add_node("mu", r"$\mu$", node_x-1, node_y-2, plot_params = {'facecolor':'lightgrey'})

# Overall variance in threshold (combination of between and within-whale variance)
pgm.add_node("omega", r"$\omega^2$", node_x, node_y-2, plot_params = {'facecolor':'lightgrey'})


# Add in the edges

pgm.add_edge("delta", "y", plot_params = grey_line)
pgm.add_edge("mu_i", "y", plot_params = solid_line)
pgm.add_edge("mu", "mu_i", plot_params = solid_line)
pgm.add_edge("omega", "mu_i", plot_params = solid_line)

# And a plate.

# label=r"$\delta$ = 2.5, ..., 40 dB"
pgm.add_plate([node_x+0.75, node_y-0.25, 0.5, 0.5], shift = -0.5, rect_params = text_label_noborder)

# Render and save

pgm.render()
pgm.savefig("dag_scenario1.png", dpi = 750)
