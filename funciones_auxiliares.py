import networkx as nx
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import gudhi

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

### Height filtration from trimesh along a direction
def height_filtration_from_mesh(mesh, direction=[1,1,1]):
    vertex_heights = np.dot(mesh.vertices, direction)
    # Get maximum of edges and triangles
    edge_heights = np.max(vertex_heights[mesh.edges_unique], axis=1)
    tri_heights = np.max(vertex_heights[mesh.faces], axis=1)
    # Create simplex tree
    st = gudhi.SimplexTree()
    # Insert vertices with heights
    vertices = np.asarray(np.arange(len(mesh.vertices)).reshape(1, -1), dtype=np.int32)
    st.insert_batch(vertices, vertex_heights)
    # Insert edges
    edges = np.array(mesh.edges_unique.T, dtype=np.int32)
    st.insert_batch(edges, edge_heights)
    # Insert triangles
    triangles = np.array(mesh.faces.T, dtype=np.int32)
    st.insert_batch(triangles, tri_heights)
    # Final safety check
    st.make_filtration_non_decreasing()
    return st
    
### Funciones de representación

def plot_simplex_tree_2D(st, pos=None, figsize=(6,6), facecolors='skyblue', alpha=0.5, with_labels=True, node_size=1000, font_size=16, seed=4, points=False):
    dict_spx = diccionario_simplices(st)
    G = nx.Graph(dict_spx[1])
    if points:
        # pos must be the np array of points
        pos = {i : pos[i] for i in range(pos.shape[0])}
    elif pos is None:
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
    # Plotting options when plotting with points
    if points:
        plt.gca().axis("on")
        plt.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True, colors='black')


def plot_simplex_tree_3D(st, points, alpha_faces=0.5, figsize=(5,5), use_filtration=True, ax=None, plot_lower_star_generators=False):
    """ Función para visualizar un complejo simplicial:
    st: complejo simplicial, estructura `simplex_tree`de Gudhi
    points: puntos en formato numpy.array (numero de puntos, 3) 
    """
    # Vamos a extraer y agrupar las aristas y los triángulos a partir de st
    vertices = []
    edges = []
    triangles = []
    triangle_filtrations = []
    for simplex, filtration in st.get_skeleton(2): # We only need up to 2D faces for 3D visualization
        dim = len(simplex) - 1
        if dim == 0:
            vertices.append(simplex[0])
        elif dim == 1:
            edges.append(simplex)
        elif dim == 2:
            triangles.append(simplex)
            triangle_filtrations.append(filtration)
            
    # Initialize the figure if axis not given
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    # Plot all Vertices in one command
    vtx_coords = points[vertices]
    ax.scatter(vtx_coords[:, 0], vtx_coords[:, 1], vtx_coords[:, 2], 
                color='black', s=10, zorder=3)
    
    # Plot all Edges in one command using Line3DCollection
    if edges:
        edge_coords = points[np.array(edges)] # Shape: (num_edges, 2, 3)
        edge_collection = Line3DCollection(edge_coords, colors='black', linewidths=0.2, alpha=0.4, zorder=2)
        ax.add_collection3d(edge_collection)
    
    # Plot all Triangles (Faces) in one command using Poly3DCollection
    if triangles:
        if use_filtration:
            tri_coords = points[np.array(triangles)] 
            filtrations_array = np.array(triangle_filtrations)
            norm = mcolors.Normalize(vmin=filtrations_array.min(), vmax=filtrations_array.max())
            face_colors = cm.plasma_r(norm(filtrations_array))
        else:
            tri_coords = points[np.array(triangles)] 
            # tomamos el centroide de cada triangulo
            centroids = np.mean(tri_coords, axis=1) 
            # tomamos las coordenadas z del centroide de cada triangulo
            z_centers = centroids[:, 2] 
            # normalizamos los centroides de 0 a 1 para el colormap
            norm = mcolors.Normalize(vmin=z_centers.min(), vmax=z_centers.max())
            # calculamos los colores de cada triangulo segun z_centers
            face_colors = cm.plasma_r(norm(z_centers))
        # 6. Pass the color array to facecolors
        tri_collection = Poly3DCollection(tri_coords, facecolors=face_colors, edgecolors='none', alpha=alpha_faces, zorder=1)
        ax.add_collection3d(tri_collection)

    # Plot lower star generators 
    if plot_lower_star_generators:
        # compute_persistence MUST be called before extracting generators in GUDHI
        st.compute_persistence()
        
        
        
        # lower_star_persistence_generators returns (regular_pairs, essential_features)
        regular_pairs, essential_features = st.lower_star_persistence_generators()
        # Loop through dimensions using the returned regular pairs
        for dim, pairs_in_dim in enumerate(regular_pairs):
            if len(pairs_in_dim) == 0:
                continue # Skip if there are no generators in this dimension
                
            # Extract birth and death vertex indices
            birth_vertices = pairs_in_dim[:, 0]
            death_vertices = pairs_in_dim[:, 1]

            # Grab the specific color for this dimension using Set1
            color = cm.Set1(dim)
            
            # Map indices to their respective 3D coordinates
            birth_coords = points[birth_vertices]
            death_coords = points[death_vertices]
            
            # Range over pairs
            for i, (b_coord, d_coord) in enumerate(zip(birth_coords, death_coords)):
                # Plot a line joining persistence pairs
                ax.plot([b_coord[0], d_coord[0]], 
                        [b_coord[1], d_coord[1]], 
                        [b_coord[2], d_coord[2]], 
                        color=color, linewidth=3, zorder=1000)

                # Legends 
                label_birth = f"Birth Dim {dim}" if i == 0 else "_nolegend_"
                label_death = f'Death Dim {dim}' if i == 0 else "_nolegend_"
                # Birth point
                ax.plot([b_coord[0]], [b_coord[1]], [b_coord[2]], 
                        marker='o', markersize=8, color=color, markeredgecolor='white', 
                        linestyle='None', zorder=1001, label=label_birth)
                
                # Death point
                ax.plot([d_coord[0]], [d_coord[1]], [d_coord[2]], 
                        marker='X', markersize=8, color=color, markeredgecolor='white', 
                        linestyle='None', zorder=1001, label=label_death)

        # Loop through dimensions using the essential features
        for dim, features_in_dim in enumerate(essential_features):
            if len(features_in_dim) == 0:
                continue # Skip if there are no generators in this dimension

            # Grab the specific color for this dimension using Set1
            color = cm.Set1(dim)
        
            # Map indices to their respective 3D coordinates
            features_coords = points[features_in_dim]
            
            # Range over pairs
            for i, coord in enumerate(features_coords):
                label_feature = f"Feature Dim {dim}" if i == 0 else "_nolegend_"
                # feature coord (plot)
                ax.plot([b_coord[0]], [b_coord[1]], [b_coord[2]], 
                        marker='s', markersize=8, color=color, markeredgecolor='white', 
                        linestyle='None', zorder=1002, label=label_feature)
                
        # Optional: You can uncomment this to show a legend for the dimensions
        # ax.legend(loc="upper right")
    # ---------------------- 
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    # 2. Force the 3D axis limits to frame the entire point cloud
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # Set aspect proportional
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    
