# # author: Tang Tiong Yew
# # email: tiongyewt@sunway.edu.my
# # Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# # Copyright 2023
# #
# import panda as pd
#
#
# def collabByRegionPlot(NetMatrix, normalize=None, n=None, degree=None, type="auto", label=True,
#                        labelsize=1, label_cex=False, label_color=False, label_n=float('inf'), halo=False,
#                        cluster="walktrap", community_repulsion=0, vos_path=None, size=3, size_cex=False,
#                        curved=False, noloops=True, remove_multiple=True, remove_isolates=False, weighted=None,
#                        edgesize=1, edges_min=0, alpha=0.5, verbose=True):
#     NetMatrix.index = NetMatrix.columns = list(map(str.lower, NetMatrix.index))
#     labelCo = pd.DataFrame({'countries': NetMatrix.index})
#     countries = pd.read_csv('countries.csv').assign(countries=lambda x: x['countries'].str.lower())
#     labelCo = labelCo.merge(countries, on='countries', how='left')
#
#     if labelCo.shape[0] < 2:
#         print("The argument NetMatrix is not a country collaboration network matrix")
#         return None
#
#     regions = labelCo['continent'].unique()
#     n_regions = len(regions)
#
#     net = {}
#     for i in regions:
#         reg_co = labelCo.query('continent == @i')['countries']
#         NetMatrix_reg = NetMatrix.loc[reg_co, reg_co]
#
#         if not NetMatrix_reg.empty:
#             net[i] = networkPlot(NetMatrix_reg, normalize=normalize, n=float('inf'), degree=degree, Title=i,
#                                  type=type, label=label, labelsize=labelsize, label_cex=label_cex,
#                                  label_color=label_color, label_n=label_n, halo=halo, cluster=cluster,
#                                  community_repulsion=community_repulsion, vos_path=vos_path, size=size,
#                                  size_cex=size_cex, curved=curved, noloops=noloops, remove_multiple=remove_multiple,
#                                  remove_isolates=remove_isolates, weighted=weighted, edgesize=edgesize,
#                                  edges_min=edges_min, alpha=alpha, verbose=False)
#             net[i]['NetMatrix_reg'] = NetMatrix_reg
#
#     if verbose:
#         l = len(net) // 2 + 1
#         fig, axs = plt.subplots(nrows=l, ncols=2, figsize=(10, l * 5))
#         axs = axs.flatten()
#         for i, net_i in enumerate(net.values()):
#             plot(net_i['graph'], ax=axs[i])
#         plt.show()
#
#     return net