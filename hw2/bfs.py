import csv
from queue import Queue
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    """
    First, I read graph data from a CSV file using csv.DictReader, storing it as an adjacency list in the 
    edge dictionary. Then initialize a dictionary 'distance' to track the shortest distance from start node 
    to each node, and a queue 'Q' for BFS traversal. Next, enqueue the start node with distance 0 into 
    Q, then iteratively dequeue nodes, update distances, and enqueue adjacent unvisited nodes until 
    the end node is reached or Q is empty. Once the traversal is complete, reconstruct the shortest 
    path from the start node to the end node using pointers stored in the distance dictionary. Finally,
    return the reconstructed path, the distance of the shortest path, and the number of visited nodes.
    """
    edge = {}
    with open(edgeFile, newline= '') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            s = int(row['start'])
            e = int(row['end'])
            d = float(row['distance'])
            if s in edge:
                edge[s].append((e, d))
            else:
                edge[s] = [(e, d)]
    distance = {start: (0, None)}
    Q = Queue()
    Q.put_nowait((start, 0))
    while not Q.empty():
        cur_node, cur_dist = Q.get_nowait()
        if cur_node not in edge:
            continue
        for next_node, d in edge[cur_node]:
            if next_node in distance:
                continue
            distance[next_node] = (cur_dist + d, cur_node)
            Q.put_nowait((next_node, cur_dist + d))
            if next_node == end:
                break
    path = [end]
    while distance[path[-1]][1] is not None:
        path.append(distance[path[-1]][1])
    path.reverse()
    dist = distance[end][0]
    num_visited = len(distance)
    return path, dist, num_visited
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
