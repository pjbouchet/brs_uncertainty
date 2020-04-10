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
pgm.add_node("y", r"$y_{ij}$", node_x, node_y)

# Process model

pgm.add_node("t", r"$t_{ij}$", node_x, node_y-1) # True underlying threshold for ith whale on jth exposure session
pgm.add_node("sigma", r"$\sigma^2$", node_x, node_y-2, plot_params = {'facecolor':'lightgrey'}) # Within-whale, between exposure variance
pgm.add_node("mu_ij", r"$\mu_{ij}$", node_x-1, node_y-1) # Expected threshold for whale i in exposure session j
pgm.add_node("mu_i", r"$\mu_{i}$", node_x-1, node_y-2) # Expected threshold for whale i
pgm.add_node("mu", r"$\mu$", node_x-1, node_y-3, plot_params = {'facecolor':'lightgrey'}) # Mean threshold for all whales
pgm.add_node("phi", r"$\phi^2$", node_x, node_y-3, plot_params = {'facecolor':'lightgrey'}) # Between-whale variance in threshold

pgm.add_node("beta", r"$\beta$", node_x-2, node_y-3, plot_params = {'facecolor':'lightgrey'}) # Coefficient for MFAS term
pgm.add_node("alpha", r"$\alpha$", node_x-3, node_y-3, plot_params = {'facecolor':'lightgrey'}) # Coefficient for exposure term

pgm.add_node("mfas", r"$I(MFAS)_{ij}$", node_x-2, node_y, shape="rectangle", aspect=2.5) # MFAS vs LFAS
pgm.add_node("expos", r"$I(exposure)_{ij}$", node_x-3, node_y-1, shape="rectangle", aspect=3) # Previously exposed vs first exposure

# Add in the edges

pgm.add_edge("delta", "y", plot_params = grey_line)

pgm.add_edge("y", "t", plot_params = solid_line)
pgm.add_edge("sigma", "t", plot_params = solid_line)
pgm.add_edge("mu_ij", "t", plot_params = solid_line)
pgm.add_edge("mu_i", "mu_ij", plot_params = grey_line)
pgm.add_edge("mu", "mu_i", plot_params = solid_line)
pgm.add_edge("phi", "mu_i", plot_params = solid_line)

pgm.add_edge("beta", "mu_ij", plot_params = grey_line)
pgm.add_edge("alpha", "mu_ij", plot_params = grey_line)

pgm.add_edge("mfas", "mu_ij", plot_params = grey_line)
pgm.add_edge("expos", "mu_ij", plot_params = grey_line)

# And a plate

pgm.add_plate([node_x-3, node_y-4, 0.5, 0.5], label=r"$\mu_{ij} = \mu_i + \alpha \cdot I(exposure) + \beta \cdot I(MFAS)$", rect_params = text_label_noborder)

# Render and save

pgm.render()
pgm.savefig("dag_scenario2.png", dpi = 750)
