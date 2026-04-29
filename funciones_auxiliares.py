import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import networkx as nx
import numpy as np

def diccionario_simplices(st):
    dict_spx = {}
    for i in range(st.dimension()+1):
        dict_spx[i] = []
    for simplex, filt in st.get_simplices():
        simplex_dim = len(simplex)-1
        dict_spx[simplex_dim].append(simplex)
    return dict_spx

def ver_simplices(st):
    dict_spx = diccionario_simplices(st)
    for dim in range(st.dimension()+1):
        print(f"Símplices en dimensión {dim}:")
        print(dict_spx[dim])

def plot_simplex_tree_2D(st, pos=None, figsize=(6,6), facecolors='skyblue', alpha=0.5, with_labels=True, node_size=1000, font_size=16):
    dict_spx = diccionario_simplices(st)
    G = nx.Graph(dict_spx[1])
    if pos is None:
        pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=figsize)
    triangles = []
    options = {
        "font_size": font_size,
        "node_size": node_size,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    if with_labels:
        options["with_labels"]=True
        
    for simplex, filt in st.get_simplices():
        if len(simplex) == 3:
            triangles.append(simplex)

    # Plot Triangles (Faces)
    if triangles:
        coords = np.array([pos[i] for i in pos.keys()])
        poly_coords = [coords[nodes] for nodes in triangles]
        face_col = PolyCollection(poly_coords, edgecolors='none', facecolors=facecolors, alpha=alpha)
        ax.add_collection(face_col) 
        
    nx.draw(G, pos=pos, **options)