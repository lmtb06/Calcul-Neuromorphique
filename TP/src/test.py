import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import random
import numpy as np
import matplotlib.transforms as transforms

# Initialize directed graph
G = nx.DiGraph()
edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
G.add_edges_from(edges)

# Position nodes using spring layout
pos = nx.spring_layout(G)

fig, ax = plt.subplots()

# Draw static graph elements (nodes, edges, labels)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos, ax=ax)
ax.set_title("Blitting Example")

# Dictionary to store dynamic mini plot lines along with their transformation and scale
lines = {}

def create_line_for_edge(edge):
    # Compute endpoints and properties of the edge
    x1, y1 = pos[edge[0]]
    x2, y2 = pos[edge[1]]
    dx, dy = x2 - x1, y2 - y1
    angle = np.arctan2(dy, dx)
    length = np.hypot(dx, dy)
    # Define a scale relative to the edge length
    scale = 0.3 * length
    # Create x-data spanning a segment of length 'scale'
    x_data = np.linspace(-scale/2, scale/2, 10)
    # Generate initial random data for the y-axis
    data = [random.random() for _ in range(10)]
    y_data = (np.array(data) - 0.5) * (scale / 2)
    # Compute the edge midpoint
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    # Create transformation: rotate and translate so that the mini plot aligns with the edge
    trans = transforms.Affine2D().rotate(angle).translate(mid_x, mid_y) + ax.transData
    # Plot the mini plot using the transformation; this returns a Line2D artist
    line, = ax.plot(x_data, y_data, transform=trans, color='red', lw=2)
    return line, trans, scale

# Create a dynamic line artist for each edge and store it
for edge in G.edges():
    line, trans, scale = create_line_for_edge(edge)
    lines[edge] = (line, trans, scale)

def init():
    # The init function returns the list of dynamic artists that will be animated.
    return [line for line, _, _ in lines.values()]

def update(num):
    # For each edge, update its dynamic line with new random data
    for edge, (line, trans, scale) in lines.items():
        x_data = np.linspace(-scale/2, scale/2, 10)
        data = [random.random() for _ in range(10)]
        y_data = (np.array(data) - 0.5) * (scale/2)
        line.set_data(x_data, y_data)
    ax.set_title(f"Time Step {num}")
    return [line for line, _, _ in lines.values()]

# Create animation with blitting enabled.
ani = animation.FuncAnimation(fig, update, init_func=init, frames=1000, interval=100, blit=True, repeat=False)
plt.show()
