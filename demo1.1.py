import numpy as np
import networkx as nx
import pandas as pd
from pandas import read_csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    f = open('passingevents.csv', encoding='UTF-8')
    f1 = open('fullevents.csv', encoding='UTF-8')
    f2 = open('matches.csv', encoding='UTF-8')
    data_passing_events = read_csv(f)
    data_full_events = read_csv(f1)
    data_matches = read_csv(f2)
except:
    print('no file in dic!')


# return the number of passes between two players according to MatchID,MatchPeriod,players'names
def num_bt_players(matchID, matchPeriod, player_name1, player_name2):
    data1 = data_passing_events[data_passing_events['OriginPlayerID'] == player_name1]
    data2 = data1[data1['MatchID'] == matchID]
    data3 = data2[data2['MatchPeriod'] == matchPeriod]
    data4 = data3[data3['DestinationPlayerID'] == player_name2]
    return data4.shape[0]


# return adjacncy matrix according to players,MatchID,MatchPeriod
def get_adjacency_matrix(players, matchID, matchPeriod):
    length = len(players)
    adjacency_matrix = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            adjacency_matrix[i, j] = num_bt_players(matchID, matchPeriod, players[i], players[j])
    return adjacency_matrix


# return size matrix according to node's degree
def set_size_by_degree(G, adjacency_matrix):
    nsize = np.zeros(adjacency_matrix.shape[0])
    for _ in range(adjacency_matrix.shape[0]):
        nsize[_] = G.degree(_)
    median = np.median(nsize)
    for _ in range(adjacency_matrix.shape[0]):
        nsize[_] = 1000 + 450 * (nsize[_] - median)
    return nsize


# return size matrix according to pagerank
def set_size_by_rank(G, adjacency_matrix):
    nsize = np.zeros(adjacency_matrix.shape[0])
    matrix = np.zeros(adjacency_matrix.shape[0])
    pagerank_list = nx.pagerank(G, alpha=1)
    for _ in range(adjacency_matrix.shape[0]):
        matrix[_] = pagerank_list[_]
    median = np.median(matrix)
    for _ in range(adjacency_matrix.shape[0]):
        nsize[_] = 10000 * (pagerank_list[_] - median) + 500
    return nsize


def draw_court():
    plt.axis('off')
    plt.plot([0, 0], [100, 0], [100, 100], [0, 100], [0, 100], [0, 0], [0, 100], [100, 100], [50, 50], [0, 100],
             [0, 4.583], [57.6, 57.6], [4.583, 4.583], [57.6, 42.4], [4.583, 0], [42.4, 42.4], [0, 13.75],
             [31.67, 31.67], [13.75, 13.75], [31.67, 68.33], [13.75, 0], [68.33, 68.33], [95.417, 95.417], [42.4, 57.6],
             [95.417, 100], [57.6, 57.6], [95.417, 100], [42.4, 42.4], [86.25, 86.25], [31.67, 68.33], [86.25, 100],
             [68.33, 68.33], [100, 86.25], [31.67, 31.67],
             color='#000000', )
    x = np.arange(42.4, 57.6, 0.0001)
    y1 = np.sqrt(7.6 ** 2 - (x - 50) ** 2) + 50
    y2 = 50 - np.sqrt(7.6 ** 2 - (x - 50) ** 2)
    plt.plot(x, y1, x, y2, color='#000000')


# def set_color_by_degree(G):
#     ncolor = ['g'] * 11
#     nsize = set_size_by_degree(G)
#     ncolor[np.argmax(nsize)] = 'r'
#     ncolor[np.argmin(nsize)] = 'y'
#     return ncolor

# return player's mean position according to MatchID,MatchPeriod
def get_average_position(matchID, matchPeriod, player_name):
    data1 = data_passing_events[data_passing_events['OriginPlayerID'] == player_name]
    data2 = data1[data1['MatchID'] == matchID]
    data3 = data2[data2['MatchPeriod'] == matchPeriod]
    data4 = data3.loc[:, 'EventOrigin_x']
    data5 = data3.loc[:, 'EventOrigin_y']
    return [data4.mean(), data5.mean()]


# cluster and return color matrix
def clustering_by_position(position):
    ncolor = ['g'] * position.shape[0]
    kmeans = KMeans(n_clusters=4, random_state=9).fit(position)
    for _ in range(position.shape[0]):
        if kmeans.labels_[_] == 0:
            ncolor[_] = 'g'
        elif kmeans.labels_[_] == 1:
            ncolor[_] = 'b'
        elif kmeans.labels_[_] == 2:
            ncolor[_] = 'y'
        elif kmeans.labels_[_] == 3:
            ncolor[_] = 'r'
    return ncolor


def get_players(MatchID, TeamID, MatchPeriod):
    players = []
    data1 = data_passing_events[data_passing_events['MatchID'] == MatchID]
    data2 = data1[data1['MatchPeriod'] == MatchPeriod]
    data3 = data2[data2['TeamID'] == TeamID]
    for i in range(data3.shape[0]):
        if (data3.iloc[i][2] in players):
            continue
        else:
            players.append(data3.iloc[i][2])
    return players


# plot passing network according to MatchID,MatchPeriod,TeamID
def plot_fig(matchID, matchPeriod, TeamID):
    h_players = get_players(matchID, TeamID, matchPeriod)
    o_players = get_players(matchID, 'Opponent1', matchPeriod)
    # 1.邻接矩阵 2.平均位置 3.设置标签 4.设置颜色
    adjacency_matrix = get_adjacency_matrix(h_players, matchID, matchPeriod)
    # 球员平均位置
    coordinates = np.zeros([len(h_players), 2])
    for i in range(len(h_players)):
        coordinates[i] = get_average_position(matchID, matchPeriod, h_players[i])
    plt.figure(1)
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.Graph())
    nlabels = []
    for _ in range(len(h_players)): nlabels.append(h_players[_][-2:])
    # 消除nan值
    a = np.array(np.where(np.isnan(coordinates)))
    for j in range(a.shape[1]): coordinates[a[0, j], a[1, j]] = -50
    ncolor = clustering_by_position(coordinates)
    pos = dict(zip(G.nodes(), coordinates))
    labels = dict(zip(G.nodes(), nlabels))
    nx.draw(G, pos, node_size=set_size_by_degree(G, adjacency_matrix), node_color=ncolor)  # 根据聚类结果分配颜色
    # nx.draw(G, pos, node_size=set_size_by_rank(G, adjacency_matrix), node_color=ncolor, node_shape='d')
    # nx.draw(G, pos, node_size=)
    nx.draw_networkx_labels(G, pos, labels, font_size=15, )
    nx.draw_networkx_edges(G, pos, width=[float(d['weight'] * 0.6) for (u, v, d) in G.edges(data=True)], edge_color='c')
    # 足球场(120m*90m)等尺度缩小
    plt.plot([0, 0], [100, 0], [100, 100], [0, 100], [0, 100], [0, 0], [0, 100], [100, 100], [50, 50], [0, 100],
             [0, 4.583],
             [57.6, 57.6], [4.583, 4.583], [57.6, 42.4], [4.583, 0], [42.4, 42.4], [0, 13.75], [31.67, 31.67],
             [13.75, 13.75], [31.67, 68.33], [13.75, 0], [68.33, 68.33],
             [95.417, 95.417], [42.4, 57.6], [95.417, 100], [57.6, 57.6], [95.417, 100], [42.4, 42.4],
             [86.25, 86.25], [31.67, 68.33], [86.25, 100], [68.33, 68.33], [100, 86.25], [31.67, 31.67],
             c='black', )
    x = np.arange(42.4, 57.6, 0.0001)
    y1 = np.sqrt(7.6 ** 2 - (x - 50) ** 2) + 50
    y2 = 50 - np.sqrt(7.6 ** 2 - (x - 50) ** 2)
    plt.plot(x, y1, x, y2, c='black')
    return adjacency_matrix, G


# return density according to adjacency matrix
def get_density(adjacency_matrix):
    x = 0
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0: x = x + 1
    return x / (adjacency_matrix.shape[0] * (adjacency_matrix.shape[0] - 1))


# plt.show()
# Matrix = []
# Matrix = [0.6454545454545455, 0.5363636363636364, 0.6818181818181818, 0.6818181818181818, 0.7090909090909091,
#           0.6818181818181818, 0.6545454545454545, 0.6636363636363637, 0.4727272727272727, 0.6636363636363637,
#           0.5727272727272728, 0.39090909090909093, 0.5151515151515151, 0.6272727272727273, 0.5454545454545454, 0.3,
#           0.6727272727272727, 0.6181818181818182, 0.6090909090909091, 0.6181818181818182, 0.7272727272727273,
#           0.5727272727272728, 0.5545454545454546, 0.6636363636363637, 0.6818181818181818, 0.5636363636363636,
#           0.5227272727272727, 0.6090909090909091, 0.5, 0.6181818181818182, 0.5909090909090909, 0.3333333333333333,
#           0.5545454545454546, 0.6363636363636364, 0.7090909090909091, 0.6909090909090909, 0.5545454545454546, 0.5]
# for i in range(1, 39):
#     adjacency_matrix = plot_fig(i, "2H")
#     Matrix.append(cal_density(adjacency_matrix))
# print(Matrix) #结果就是上面那个
# Matrix = [0.47435897435897434, 0.3269230769230769, 0.40384615384615385, 0.6136363636363636, 0.521978021978022,
#           0.38461538461538464, 0.5384615384615384, 0.3717948717948718, 0.45454545454545453, 0.553030303030303,
#           0.3333333333333333, 0.43636363636363634, 0.46794871794871795, 0.44505494505494503, 0.489010989010989,
#           0.2727272727272727, 0.358974358974359, 0.5, 0.5909090909090909, 0.36813186813186816, 0.5606060606060606,
#           0.36813186813186816, 0.4423076923076923, 0.44871794871794873, 0.41025641025641024, 0.5576923076923077,
#           0.44696969696969696, 0.42948717948717946, 0.48717948717948717, 0.3956043956043956, 0.45054945054945056,
#           0.27564102564102566, 0.36538461538461536, 0.4230769230769231, 0.4166666666666667, 0.3901098901098901,
#           0.4935897435897436, 0.5]
# x = np.linspace(1, 39, 38)
# plt.figure(2)
# plt.plot(x, Matrix, marker='o', ls='--', c='#000000')
# plt.xlim(1, 39)
# plt.ylim(0, 1)
# plt.title('Network density of 38 games (second half)')
# plt.show()
# # 射门数量比较
# data = data_full_events[data_full_events['MatchID'] == 1]
# data1 = data[data['EventType'] == 'Shot']
# print(data1.shape[0])
# data2 = data[data['TeamID'] == 'Huskies']
# data3 = data2[data2['EventType'] == 'Shot']
# data4 = data[data['TeamID'] == 'Opponent1']
# data5 = data4[data4['EventType'] == 'Shot']
# print(data3.shape[0])
# print(data5.shape[0])
# # name_list = ['Monday','Tuesday','Friday','Sunday']
# # num_list = [1.5,0.6,7.8,6]
# plt.bar(range(2), [data3.shape[0], data5.shape[0]], color="#87CEFA", tick_label=['Huskies', 'Opponent1'], width=0.5)
# plt.title('The number of shots in match1')
# plt.show()

# free kick 任意球 duel 对抗 pass 传球 trow in 掷界外球
# goal kick 球门球

# 根据三个点计算三角形质心

# time = []
# for i in range(data2.shape[0]):
#     time.append(data2.iloc[i][5] / 60)
# print(time)
# for i in range(data2.shape[0]):

# return centroid according to MatchId,TeamID,MatchPeriod
def get_centroid(MatchID, TeamID, MatchPeriod):
    data = data_passing_events[data_passing_events['MatchID'] == MatchID]
    data1 = data[data['TeamID'] == TeamID]
    data2 = data1[data1['MatchPeriod'] == MatchPeriod]
    j = 0
    centroid = []
    for i in range(data2.shape[0] // 3):
        point1 = data2.iloc[j][7]
        point2 = data2.iloc[j + 1][7]
        point3 = data2.iloc[j + 2][7]
        centroid.append((point1 + point2 + point3) / 3)
        j = j + 3
    if data2.shape[0] % 3 == 1:
        point = data2.iloc[j][7]
        centroid.append(point)
    elif data2.shape[0] % 3 == 2:
        point1 = data2.iloc[j][7]
        point2 = data2.iloc[j + 1][7]
        centroid.append((point1 + point2) / 2)
    print(centroid)
    time = []
    for i in range(data2.shape[0]):
        time.append(data2.iloc[i][5] / 60)
    return centroid


# retrun attack_ratio accoring to MatchID,TeamID
def get_attack_ratio(MatchId, TeamID):
    data = data_passing_events[data_passing_events['TeamID'] == TeamID]
    data1 = data[data['MatchID'] == MatchId]
    data2 = data1.loc[data1['EventOrigin_x'] > 50]
    attack_ratio = data2.shape[0] / data1.shape[0]
    return attack_ratio


def get_scores(MatchID):
    data = data_matches[data_matches['MatchID'] == MatchID]
    # return np.exp(data.iloc[0][3])
    return data.iloc[0][3]


def get_result(MatchID):
    data = data_matches[data_matches['MatchID'] == MatchID]
    if data.iloc[0][2] == 'win':
        return 1
    elif data.iloc[0][2] == 'tie':
        return 0
    elif data.iloc[0][2] == 'loss':
        return -1


def get_event(MatchID, EventType):
    data = data_full_events[data_full_events['MatchID'] == MatchID]
    data1 = data[data['TeamID'] == 'Huskies']
    data2 = data1[data1['EventType'] == EventType]
    return data2.shape[0]


def get_total_passes(MatchID):
    data = data_passing_events[data_passing_events['MatchID'] == MatchID]
    data1 = data[data['TeamID'] == 'Huskies']
    return data1.shape[0]


# correlations
# x = []
# y = []
# z = []
# z1 = []
# for _ in range(1, 39):
#     x.append(get_result(_))
#     y.append(get_event(_, 'Save attempt'))
#     z.append(get_event(_, 'Goalkeeper leaving line'))
#     z1.append(get_event(_, 'Duel'))
# # plt.scatter(x, y)
# # df = pd.DataFrame({'x': x, 'y': y})
# print(y)
# print(x)
# print(z1)
# # print(df.corr())
# x1 = np.linspace(1, 39, 38)
# # y1 = np.log(5 / np.array(y) + np.array(z) / 10  + np.array(z1) / 1000)
# y1 = [0.75, 0, 1, 1, 1, 1, 1, 0, -1, -1, 1, 0, -1, 1, 1, 0, 1, 1, 0, 0, -1, -1, -1, 0, 1, -1, 1, -1, -1, 1, 1, -1, 0, 0,
#       1, 1, 0, -1]
# pd.set_option('display.max_columns',1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth',1000)
# print(y1)
# plt.figure()
# plt.scatter(x1, y1)
# df = pd.DataFrame({r'\mu': y, 'Goalkeeper leaving line': z, 'Duel': z1, 'H': y1, 'Result': x})
# print(df.corr())

# plt.figure(1)
# centroid_1_h = get_centroid(1, 'Huskies', '1H') + get_centroid(1, 'Huskies', '2H')
# centroid_1_o = get_centroid(1, 'Opponent1', '1H') + get_centroid(1, 'Opponent1', '2H')
# time_1_h = np.linspace(0, 90, len(centroid_1_h))
# time_1_o = np.linspace(0, 90, len(centroid_1_o))
# plt.plot(time_1_h, np.array(centroid_1_h))
# plt.plot(time_1_o, np.array(centroid_1_o))
# plt.ylabel('X')
# plt.xlabel('Time(minutes)')
# plt.title('Huskies VS Opponent1 in Match1')
# plt.legend(['Huskies', 'Opponent1'])
# plt.figure(2)
# centroid_1_h = get_centroid(31, 'Huskies', '1H') + get_centroid(31, 'Huskies', '2H')
# centroid_1_o = get_centroid(31, 'Opponent1', '1H') + get_centroid(31, 'Opponent1', '2H')
# time_1_h = np.linspace(0, 90, len(centroid_1_h))
# time_1_o = np.linspace(0, 90, len(centroid_1_o))
# plt.plot(time_1_h, np.array(centroid_1_h))
# plt.plot(time_1_o, np.array(centroid_1_o))
# plt.ylabel('X')
# plt.xlabel('Time(minutes)')
# plt.title('Huskies VS Opponent1 in Match31')
# plt.legend(['Huskies', 'Opponent1'])
# plt.show()

# attack_ratio = []
# for i in range(1, 39):
#     attack_ratio.append(get_attack_ratio(i, 'Huskies'))
# # y = np.array(attack_ratio)
# # x = np.linspace(1, 39, 38)
# # plt.plot(x, y)
# # plt.title('Attack Ratio of Huskies in 38 Matches')
# # plt.ylim(0, 1)
# # plt.xlabel('MatchID')
# # plt.ylabel('Attack Ratio')
# # plt.show()
# attack_ratio_o1 = (attack_ratio[0] + attack_ratio[30]) / 2
# attack_ratio_o2 = (attack_ratio[1] + attack_ratio[31]) / 2
# attack_ratio_o3 = (attack_ratio[2] + attack_ratio[18]) / 2
# attack_ratio_o4 = (attack_ratio[3] + attack_ratio[22]) / 2
# attack_ratio_o5 = (attack_ratio[4] + attack_ratio[21]) / 2
# attack_ratio_o6 = (attack_ratio[5] + attack_ratio[20]) / 2
# attack_ratio_o7 = (attack_ratio[6] + attack_ratio[28]) / 2
# attack_ratio_o8 = (attack_ratio[7] + attack_ratio[29]) / 2
# attack_ratio_o9 = (attack_ratio[8] + attack_ratio[25]) / 2
# attack_ratio_o10 = (attack_ratio[9] + attack_ratio[24]) / 2
# attack_ratio_o11 = (attack_ratio[10] + attack_ratio[27]) / 2
# attack_ratio_o12 = (attack_ratio[11] + attack_ratio[26]) / 2
# attack_ratio_o13 = (attack_ratio[12] + attack_ratio[36]) / 2
# attack_ratio_o14 = (attack_ratio[13] + attack_ratio[37]) / 2
# attack_ratio_o15 = (attack_ratio[14] + attack_ratio[35]) / 2
# attack_ratio_o16 = (attack_ratio[15] + attack_ratio[32]) / 2
# attack_ratio_o17 = (attack_ratio[16] + attack_ratio[34]) / 2
# attack_ratio_o18 = (attack_ratio[17] + attack_ratio[33]) / 2
# attack_ratio_o19 = (attack_ratio[18] + attack_ratio[23]) / 2
#
# attack_ratio_average = [attack_ratio_o1, attack_ratio_o2, attack_ratio_o3, attack_ratio_o4, attack_ratio_o5,
#                         attack_ratio_o6,
#                         attack_ratio_o7, attack_ratio_o8, attack_ratio_o9, attack_ratio_o10, attack_ratio_o11,
#                         attack_ratio_o12,
#                         attack_ratio_o13, attack_ratio_o14, attack_ratio_o15, attack_ratio_o16, attack_ratio_o17,
#                         attack_ratio_o18,
#                         attack_ratio_o19]
#
# sort = [14, 19, 18, 15, 4, 9, 3, 13, 5, 1, 12, 8, 7, 10, 11, 2, 16, 17, 6]
#
# plt.figure(1)
# for _ in range(19):
#     plt.scatter(sort[_], attack_ratio[_], c='c')
#     plt.scatter(sort[_], attack_ratio_average[_], c='#000000', marker='^')
#
# plt.scatter(sort[2], attack_ratio[18], c='c')
# plt.scatter(sort[18], attack_ratio[19], c='c')
# plt.scatter(sort[5], attack_ratio[20], c='c')
# plt.scatter(sort[4], attack_ratio[21], c='c')
# plt.scatter(sort[3], attack_ratio[22], c='c')
# plt.scatter(sort[18], attack_ratio[23], c='c')
# plt.scatter(sort[9], attack_ratio[24], c='c')
# plt.scatter(sort[8], attack_ratio[25], c='c')
# plt.scatter(sort[11], attack_ratio[26], c='c')
# plt.scatter(sort[10], attack_ratio[27], c='c')
# plt.scatter(sort[6], attack_ratio[28], c='c')
# plt.scatter(sort[7], attack_ratio[29], c='c')
# plt.scatter(sort[0], attack_ratio[30], c='c')
# plt.scatter(sort[1], attack_ratio[31], c='c')
# plt.scatter(sort[15], attack_ratio[32], c='c')
# plt.scatter(sort[17], attack_ratio[33], c='c')
# plt.scatter(sort[16], attack_ratio[34], c='c')
# plt.scatter(sort[14], attack_ratio[35], c='c')
# plt.scatter(sort[12], attack_ratio[36], c='c')
# plt.scatter(sort[13], attack_ratio[37], c='c')
# x = np.linspace(0.5, 19.5)
# y = 0.54 - 0.0088 * x
# plt.plot(x, y, linestyle='--', c='r')
# plt.ylim(0.2, 0.8)
# plt.ylabel('attack_defense ratio', fontsize=13)
# plt.title('Opponents’ Counter-Strategies Influence', fontsize=12)
# plt.xticks(np.linspace(1, 20, 19),
#            ('O14', 'O19', 'O18', 'O15', 'O4', 'O9', 'O3', 'O13',
#             'O5', 'O1', 'O12', 'O8', 'O7', 'O10', 'O11', 'O2',
#             'O16', 'O17', 'O6',), rotation=60)
# plt.legend(['trend of ratio', 'ratio of each match', 'average ratio'], fontsize=13)
# plt.show()

# Matrix1 = []
# Matrix2 = []
# sum1 = 0
# sum2 = 0
# # for i in range(1, 3):
# #     adjacency_matrix1 = plot_fig(i, "1H", 'Huskies')
# #     adjacency_matrix2 = plot_fig(i, "2H", 'Huskies')
# #     Matrix1.append(get_density(adjacency_matrix1))
# #     Matrix2.append(get_density(adjacency_matrix2))
# # for i in Matrix1:
# #     sum1 = sum1 + i
# # for j in Matrix2:
# #     sum2 = sum2 + j
# # print((sum1 / len(Matrix1) + sum2 / len(Matrix2)) / 2)
#
# adjacency_matrix1 = plot_fig(20, "1H", 'Opponent19')
# adjacency_matrix2 = plot_fig(20, "2H", 'Opponent19')
# adjacency_matrix3 = plot_fig(24, "2H", 'Opponent19')
# adjacency_matrix4 = plot_fig(24, "2H", 'Opponent19')
# Matrix1.append(get_density(adjacency_matrix1))
# Matrix1.append(get_density(adjacency_matrix2))
# Matrix2.append(get_density(adjacency_matrix3))
# Matrix2.append(get_density(adjacency_matrix4))
# for i in Matrix1:
#     sum1 = sum1 + i
# for j in Matrix2:
#     sum2 = sum2 + j
# print((sum1 / len(Matrix1) + sum2 / len(Matrix2)) / 2)

# H 0.4958 O1 0.4537 O2 0.6835 O3 0.5230 O4 0.5785 O5 0.5700 O6 0.4899 O7 0.5023 O8 0.4885 O9 0.6048 O10 0.4554 O11 0.4366 O12 0.6028 O13 0.4515 O14 0.4482 O15 0.5115 O16 0.5669 O17 0.5306 O18 0.4898 O19 0.4126

#           win tie loss
# Coach1     2   2   5
# Coach2     2   1   2
# Coach3     9   7   8
# label_list = ['Coach1', 'Coach2', 'Coach3']
# num_list1 = [2, 2, 9]
# num_list2 = [2, 1, 7]
# num_list3 = [5, 2, 8]
#
# plot_fig(1, '1H', 'Huskies')
# plt.show()

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# plot shot position according to MatchID and return names of men who shot
def plot_shot_position(MatchID):
    shot_x = []
    shot_y = []
    shot_m = []
    data = data_full_events[data_full_events['EventType'] == 'Shot']
    data1 = data[data['MatchID'] == MatchID]
    data2 = data1[data1['TeamID'] == 'Huskies']
    for _ in range(data2.shape[0]):
        shot_x.append(data2.iloc[_][8])
        shot_y.append(data2.iloc[_][9])
        shot_m.append(data2.iloc[_][2])
    # draw_court()
    for _ in range(len(shot_x)):
        plt.text(shot_x[_], shot_y[_], shot_m[_][-2:])
    return shot_m


# plot the start and end position of a samet pass and return the total distance of smart passes according to MatchID,TeamID
def plot_smart_pass_position(MatchID, TeamID):
    start_x = []
    start_y = []
    end_x = []
    end_y = []
    m = []
    data = data_full_events[data_full_events['EventSubType'] == 'Smart pass']
    data1 = data[data['MatchID'] == MatchID]
    data2 = data1[data1['TeamID'] == TeamID]
    for _ in range(data2.shape[0]):
        start_x.append(data2.iloc[_][8])
        start_y.append(data2.iloc[_][9])
        end_x.append(data2.iloc[_][10])
        end_y.append(data2.iloc[_][11])
        m.append(data2.iloc[_][2])
    distance = np.sqrt((np.array(start_x) - np.array(end_x)) ** 2 + (np.array(start_y) - np.array(end_y)) ** 2)
    # if len(start_x) == 0:
    #     average_distance = 0
    # else:
    #     average_distance = sum(distance) / (len(start_x))
    draw_court()
    for _ in range(len(start_x)):
        plt.scatter(start_x[_], start_y[_])
        plt.scatter(end_x[_], end_y[_])
        plt.arrow(start_x[_], start_y[_], end_x[_] - start_x[_], end_y[_] - start_y[_], width=1,
                  length_includes_head=True, fc='c')
    for _ in range(len(start_x)):
        plt.text(start_x[_], start_y[_], m[_][-2:], fontsize=13)
    # return average_distance
    return sum(distance)


def draw_circle():
    plt.axis('off')
    # plt.axis('equal')
    plt.plot([50, 50], [np.sqrt(7.6 ** 2 - (50 - 50) ** 2) + 50, 50 - np.sqrt(7.6 ** 2 - (50 - 50) ** 2)], c='black')
    x = np.arange(42.4, 57.6, 0.0001)
    y1 = np.sqrt(7.6 ** 2 - (x - 50) ** 2) + 50
    y2 = 50 - np.sqrt(7.6 ** 2 - (x - 50) ** 2)
    plt.plot(x, y1, x, y2, c='black')


# m_dict = {}
# for _ in range(38):
#     m = plot_shot_position(_ + 1)
#     for key in m:
#         m_dict[key] = m_dict.get(key, 0) + 1
#
# print(m_dict)
# x = []
# y = []
# for key in m_dict.keys(): x.append(key)
# for value in m_dict.values(): y.append(value)
# plt.barh(range(27), y, height=0.7, color='steelblue', alpha=0.5)
# plt.yticks(range(27), x)
# for x, y in enumerate(y):
#     plt.text(y+0.2, x-0.2, '%s'%y)
# plt.xlabel('Number of Shots')
# plt.show()

# print(plot_smart_pass_position(29))
# ad = []
# x = []
# for _ in range(38):
#     ad.append(plot_smart_pass_position(_ + 1, 'Huskies'))
#     x.append(get_result(_ + 1))
#     # x.append(get_scores(_+1))
# print(ad)
# plt.plot(np.linspace(1, 39, 38), ad)
# # plt.show()
# #
# df = pd.DataFrame({'x': x, 'y': ad})
# print(df.corr())  # 胜负相关性0.169849 0.35 进球相关性-0.04 0.15

# data = data_full_events[data_full_events['OriginPlayerID'] == 'Huskies_F1']
# data1 = data[data['MatchID'] == 1]
# print(data1)
# draw_court()
# for _ in range(len(data1)):
#     plt.plot([data1.iloc[_][8], data1.iloc[_][10]], [data1.iloc[_][9], data1.iloc[_][11]])
#
# plt.show()

# A
# data = data_matches[data_matches['OpponentID'] == 'Opponent19']
# print(data)

# return attack index according to CoachID and plot
def get_attack_index(CoachID):
    if CoachID == 'Coach1':
        MatchID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif CoachID == 'Coach2':
        MatchID = [10, 11, 12, 13, 14]
    elif CoachID == 'Coach3':
        MatchID = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    pass_d = 0
    pass_f = 0
    pass_m = 0
    xd = []
    yd = []
    xf = []
    yf = []
    xm = []
    ym = []
    for _ in MatchID:
        data = data_passing_events[data_passing_events['MatchID'] == _]
        data1 = data[data['TeamID'] == 'Huskies']
        for i in range(data1.shape[0]):
            name = data1.iloc[i][2]
            if name[:9] == 'Huskies_F':
                pass_f = pass_f + 1
                xf.append(data1.iloc[i][7])
                yf.append(data1.iloc[i][8])
            elif name[:9] == 'Huskies_D':
                pass_d = pass_d + 1
                xd.append(data1.iloc[i][7])
                yd.append(data1.iloc[i][8])
            elif name[:9] == 'Huskies_M':
                pass_m = pass_m + 1
                xm.append(data1.iloc[i][7])
                ym.append(data1.iloc[i][8])
    print(pass_d, pass_m, pass_f)
    massive_d = pass_d / (pass_d + pass_m + pass_f)
    massive_f = pass_f / (pass_d + pass_m + pass_f)
    massive_m = pass_m / (pass_d + pass_m + pass_f)
    xd_average = np.mean(xd)
    xf_average = np.mean(xf)
    xm_average = np.mean(xm)
    yd_average = np.mean(yd)
    yf_average = np.mean(yf)
    ym_average = np.mean(ym)
    centroid_x = xd_average * massive_d + xf_average * massive_f + xm_average * massive_m
    centroid_y = yd_average * massive_d + yf_average * massive_f + ym_average * massive_m
    draw_circle()
    plt.plot([xd_average, xf_average], [yd_average, yf_average], [xf_average, xm_average], [yf_average, ym_average],
             [xm_average, xd_average], [ym_average, yd_average], c='lightblue')
    plt.scatter(xd_average, yd_average, c='#000000', )
    plt.scatter(xm_average, ym_average, c='#000000', )
    plt.scatter(xf_average, yf_average, c='#000000', )
    plt.scatter(centroid_x, centroid_y, c='r')
    plt.plot([xd_average, centroid_x], [yd_average, centroid_y], [xf_average, centroid_x], [yf_average, centroid_y],
             [xm_average, centroid_x], [ym_average, centroid_y], linestyle='--', c='lightblue')
    plt.text(xd_average - .3, yd_average + .3, 'D' + CoachID[-1:], fontsize=13)
    plt.text(xf_average - .3, yf_average + .3, 'F' + CoachID[-1:], fontsize=13)
    plt.text(xm_average - .3, ym_average + .3, 'M' + CoachID[-1:], fontsize=13)
    plt.text(centroid_x - .3, centroid_y + .3, 'O' + CoachID[-1:], fontsize=13)
    plt.arrow(50, 50, centroid_x - 50, centroid_y - 50, width=0.1,
              length_includes_head=True, fc='r', ec='lightblue')
    plt.scatter(50, 50, c='black')
    plt.text(50 - .3, 50 + .3, 'O', fontsize=13)
    return centroid_x, centroid_y, centroid_x / 100

# plt.subplot(1, 3, 1)
# print(get_attack_index('Coach1'))
# plt.subplot(1, 3, 2)
# print(get_attack_index('Coach2'))
# plt.subplot(1, 3, 3)
# print(get_attack_index('Coach3'))
# plt.show()
