

def diccionario_simplices(st):
    dict_spx = {}
    for i in range(st.dimension()+1):
        dict_spx[i] = []
    for simplex, filt in st.get_simplices():
        simplex_dim = len(simplex)-1
        dict_spx[simplex_dim].append(simplex)
    return dict_spx