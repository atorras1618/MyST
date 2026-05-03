import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import networkx as nx
import numpy as np
import sympy as sp

def diccionario_simplices(st):
    dict_spx = {}
    for i in range(st.dimension()+1):
        dict_spx[i] = []
    for simplex, filt in st.get_simplices():
        simplex_dim = len(simplex)-1
        dict_spx[simplex_dim].append(simplex)
    return dict_spx

def lista_simplices(st):
    dict_spx = diccionario_simplices(st)
    lista_spx = []
    for dim in range(st.dimension()+1):
        lista_spx += dict_spx[dim]
    return lista_spx

def característica_euler(st):
    dict_spx = diccionario_simplices(st)
    euler_chi = 0
    for dim in range(st.dimension()+1):
        euler_chi += ((-1)**dim) * len(dict_spx[dim])
    return euler_chi

def ver_simplices(st):
    dict_spx = diccionario_simplices(st)
    for dim in range(st.dimension()+1):
        print(f"Símplices en dimensión {dim}:")
        print(dict_spx[dim])

#### Funciones de matrices 

# Calculamos el diccionario de simplices indexado por dimensiones
def diferenciales(st):
    """ Devuelve los diferenciales de un "simplex tree" como matrices según el paquete de álgebra simbólica de python simpy
    """
    lista_dif = [[]]
    dict_spx = diccionario_simplices(st)
    for dim in range(1,st.dimension()+1):
        nspx_d = len(dict_spx[dim]) # Número símplices dimensión dim
        nspx_dm = len(dict_spx[dim-1]) # Número símplices dimensión dim -1
        # Inicializamos la matriz del diferencial como una matriz nula
        diferencial = np.zeros((nspx_dm, nspx_d)) 
        for i, spx in enumerate(dict_spx[dim]):
            for j in range(len(spx)):
                cara = spx[:j] + spx[j+1:]
                idx_cara = dict_spx[dim-1].index(cara)
                coef = (-1)**j
                diferencial[idx_cara, i] = coef
    
        lista_dif.append(sp.Matrix(diferencial.astype(int)))
    # for 
    return lista_dif
    
### Funciones de representación

def plot_simplex_tree_2D(st, pos=None, figsize=(6,6), facecolors='skyblue', alpha=0.5, with_labels=True, node_size=1000, font_size=16, seed=4):
    dict_spx = diccionario_simplices(st)
    G = nx.Graph(dict_spx[1])
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)
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

