import csv
import queue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    """
    First, I convert the start and end nodes to strings and initialize dictionaries to store graph edges 
    and heuristic values. It reads edge and heuristic data from CSV files. The edge data represents the 
    graph's connectivity, while the heuristic data provides estimated distances from each node to the 
    goal node. Then, I initialize a priority queue ‘pq’ to store nodes based on their priority, which is 
    determined by the sum of the current distance and the heuristic estimate to the goal node. The 
    start node is added to the priority queue with a priority of 0. The main loop dequeues nodes from 
    the priority queue until it reaches the end node or the priority queue becomes empty. For each 
    dequeued node, it explores its adjacent nodes. If a shorter path to an adjacent node is found, I
    update the distance in the distance dictionary and enqueue the adjacent node with its updated 
    priority into the priority queue. After finding the shortest path or exploring all nodes, I reconstruct
    the shortest path from the end node to the start node using pointers stored in the distance
    dictionary and then reverse it to ensure the correct order. Finally, it returns the shortest path, the 
    distance of the shortest path, and the number of visited nodes.
    """
    start = str(start)
    end = str(end)
    edge = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for row in rows:
            s = row[0]
            e = row[1]
            d = float(row[2])
            h = float(row[3])
            if s in edge:
                edge[s][e] = (d, h)
            else:
                edge[s] = {e: (d, h)}
    heuristic = {}
    with open(heuristicFile, newline='') as csvfile:
        rows = csv.reader(csvfile)
        title = next(rows)
        for row in rows:
            node = row[0]
            heuristic[node] = {title[i]: float(row[i]) for i in range(1, len(row))}
    pq = queue.PriorityQueue()
    pq.put((heuristic[start][end], start, -1, 0))
    From = {}
    distance = {}
    path = []
    dist = 0
    num_visited = 0
    while not pq.empty():
        now = pq.get()
        if now[1] in From:
            continue
        distance[now[1]] = now[0]
        From[now[1]] = now[2]
        num_visited += 1
        if now[1] == end:
            break
        if now[1] not in edge:
            continue
        for i in edge[now[1]]:
            temp = now[3] + edge[now[1]][i][0] + edge[now[1]][i][1]
            if i not in distance or temp < distance[i]:
                pq.put((temp, i, now[1], now[3] + edge[now[1]][i][0]))
                distance[i] = temp
    now = end
    while From[now] != -1:
        dist += edge[From[now]][now][0]
        path.append(int(now))
        now = From[now]
    path.append(int(now))
    path.reverse()
    return path, dist, num_visited
    #raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
