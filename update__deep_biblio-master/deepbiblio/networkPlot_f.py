# # author: Tang Tiong Yew
# # email: tiongyewt@sunway.edu.my
# # Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# # Copyright 2023
# #
# import numpy as np
# import igraph as ig
# import panda as pd
#
#
# def networkPlot(NetMatrix, normalize=None, n=None, degree=None, Title="Plot", type="auto", label=True, labelsize=1,
#                 label.
#
#
#     cex = False, label.color = False, label.n = None, halo = False, cluster = "walktrap", community.repulsion = 0.1, vos.path = None, size = 3, size.cex = False, curved = False, noloops = True, remove.multiple = True, remove.isolates = False, weighted = None, edgesize = 1, edges.min = 0, alpha = 0.5, verbose = True):
#
#     S = None
#     NetMatrix.columns = NetMatrix.index = NetMatrix.columns.str.lower()
#     bsk.S = True
#     l = None
#     net_groups = []
#
#     if normalize is not None:
#         S = normalizeSimilarity(NetMatrix, type=normalize)
#         bsk.S = ig.Graph.Adjacency(S, mode="UNDIRECTED", weighted=True)
#
#     if alpha < 0 or alpha > 1:
#         alpha = 0.5
#
#     # Create igraph object
#     bsk.network = ig.Graph.Adjacency(NetMatrix, mode="UNDIRECTED", weighted=weighted)
#
#     # vertex labels
#     bsk.network.vs["name"] = NetMatrix.columns
#
#     # node degree plot
#     deg = bsk.network.degree(mode="ALL")
#     deg_dist = pd.DataFrame({"node": NetMatrix.columns, "degree": deg})
#     deg_dist = deg_dist.sort_values("degree", ascending=False)
#     deg_dist["degree"] = deg_dist["degree"] / max(deg_dist["degree"])
#
#     # Compute node degrees (#links) and use that to set node size:
#     deg = np.sum(net_matrix, axis=1)
#     V(bsk.network)['deg'] = deg
#     if size_cex:
#         V(bsk.network)['size'] = (deg / np.max(deg)) * size
#     else:
#         V(bsk.network)['size'] = np.repeat(size, len(V(bsk.network)))
#
#     # Label size
#     if label_cex:
#         lsize = np.log(1 + (deg / np.max(deg))) * labelsize
#         lsize[lsize < 0.5] = 0.5  ### min labelsize is fixed to 0.5
#         V(bsk.network)['label.cex'] = lsize
#     else:
#         V(bsk.network)['label.cex'] = labelsize
#
#     # Select number of vertices to plot
#     if degree is not None:
#         Deg = deg - np.diag(net_matrix)
#         Vind = Deg < degree
#         if np.sum(~Vind) == 0:
#             print("\ndegree argument is too high!\n\n")
#             return
#         bsk.network = delete.vertices(bsk.network, which(Vind))
#         if not bsk.S:
#             bsk.S = delete.vertices(bsk.S, which(Vind))
#     elif n is not None:
#         if n > net_matrix.shape[0]:
#             n = net_matrix.shape[0]
#         nodes = np.argsort(deg)[::-1][:n]
#
#         bsk.network = delete.vertices(bsk.network,
#                                       which(~np.in1d(V(bsk.network)['name'], nodes)))
#         if not bsk.S:
#             bsk.S = delete.vertices(bsk.S,
#                                     which(~np.in1d(V(bsk.S)['name'], nodes)))
#
#     # Remove loops and multiple edges
#     if edges_min > 1:
#         remove_multiple = False
#     bsk.network = simplify(bsk.network,
#                            remove_multiple=remove_multiple,
#                            remove_loops=noloops)
#     if not bsk.S:
#         bsk.S = simplify(bsk.S,
#                          remove_multiple=remove_multiple,
#                          remove_loops=noloops)
#
#     ### graph to write in pajek format ###
#     bsk.save = bsk.network
#     V(bsk.save)['id'] = V(bsk.save)['name']
# return net
#
#
# def delete_isolates(graph, mode='all'):
#     isolates = [name for name in graph.vs if graph.degree(name, mode=mode) == 0]
#     graph.delete_vertices(isolates)
#     return graph
#
# def switchLayout(bsk_network, type, community_repulsion):
#     if community_repulsion > 0:
#         community_repulsion = round(community_repulsion * 100)
#         row = bsk_network.get_edgelist()
#         membership = bsk_network.vs["community"]
#         membership_names = dict(zip(bsk_network.vs["name"], membership))
#
#         if not bsk_network.es["weight"]:
#             bsk_network.es["weight"] = [np.apply(row, 1, weight_community, membership_names, community_repulsion, 1)]
#         else:
#             bsk_network.es["weight"] = [
#                 bsk_network.es["weight"][i] + np.apply(row[i], 1, weight_community, membership_names, community_repulsion,
#                                                     1) for i in range(len(bsk_network.es))]
#
#     switcher = {
#         "auto": lambda: bsk_network.layout_auto(),
#         "circle": lambda: bsk_network.layout_circle(),
#         "star": lambda: bsk_network.layout_star(),
#         "sphere": lambda: bsk_network.layout_sphere(),
#         "mds": lambda: bsk_network.layout_mds(),
#         "fruchterman": lambda: bsk_network.layout_fruchterman_reingold(),
#         "kamada": lambda: bsk_network.layout_kamada_kawai()
#     }
#
#     # Get the function from switcher dictionary
#     func = switcher.get(type, lambda: bsk_network.layout_auto())
#
#     # Execute the function
#     l = func()
#
#     l = bsk_network.layout_normalize(l)
#
#     layout_results = {"l": l, "bsk.network": bsk_network}
#     return layout_results
#
# def clusteringNetwork(bsk_network, cluster):
#     colorlist = ig.drawing.colors.ClusterColoringPalette(len(bsk_network))
#
#     if cluster == 'none':
#         net_groups = {'membership': [1] * len(bsk_network.vs)}
#     elif cluster == 'optimal':
#         net_groups = ig.GraphBase.community_optimal_modularity(bsk_network)
#     elif cluster == 'leiden':
#         net_groups = bsk_network.community_leiden(objective_function="modularity", n_iterations=3,
#                                                   resolution_parameter=0.75)
#     elif cluster == 'louvain':
#         net_groups = bsk_network.community_multilevel()
#     elif cluster == 'fast_greedy':
#         net_groups = bsk_network.community_fastgreedy().as_clustering()
#     elif cluster == 'leading_eigen':
#         net_groups = bsk_network.community_leading_eigenvector()
#     elif cluster == 'spinglass':
#         net_groups = bsk_network.community_spinglass()
#     elif cluster == 'infomap':
#         net_groups = bsk_network.community_infomap()
#     elif cluster == 'edge_betweenness':
#         net_groups = bsk_network.community_edge_betweenness()
#     elif cluster == 'walktrap':
#         net_groups = bsk_network.community_walktrap().as_clustering()
#     else:
#         print("\nUnknown cluster argument. Using default algorithm\n")
#         net_groups = bsk_network.community_walktrap().as_clustering()
#
#     bsk_network.vs['color'] = colorlist.get_many(net_groups.membership)
#
#     ### set edge intra-class colors
#     bsk_network.vs['community'] = net_groups.membership
#     el = bsk_network.get_edgelist()
#     E_color = []
#     for e in el:
#         if bsk_network.vs['community'][e[0]] == bsk_network.vs['community'][e[1]]:
#             C = colorlist[bsk_network.vs['community'][e[0]]]
#         else:
#             C = 'gray70'
#         E_color.append(C)
#     bsk_network.es['color'] = E_color
#     bsk_network.es['lty'] = 1
#     for i, e in enumerate(bsk_network.es):
#         if e['color'] == 'gray70':
#             bsk_network.es[i]['lty'] = 5
#     ### end
#
#     cl = {}
#     cl['bsk_network'] = bsk_network
#     cl['net_groups'] = net_groups
#     return cl
#
# def weight_community(row, membership, weigth_within, weight_between):
#     m1 = int(membership[list(np.where(np.array(list(membership.keys())) == row[0])[0])[0]])
#     m2 = int(membership[list(np.where(np.array(list(membership.keys())) == row[1])[0])[0]])
#     if m1 == m2:
#         weight = weigth_within
#     else:
#         weight = weight_between
#     return(weight)