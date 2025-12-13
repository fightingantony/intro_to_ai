import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
    """
    First, I read edge information from a file, where each edge contains data such as start node, end 
    node, distance, and speed limit. It calculates the time required to traverse each edge based on the 
    given speed limit. Then, I read heuristic information from another file, providing an estimated cost 
    from each node to the end node, considering the maximum speed limit. Using this data, and 
    applied A* algorithm. I maintain a priority queue (heap) to explore nodes based on their estimated 
    total cost, which is the sum of the current time, heuristic cost, and the time to reach the next 
    node. It iteratively explores nodes, updating the shortest path and time to reach each node as it 
    progresses. It stops when the end node is reached or when there are no more nodes to explore.
    Finally, I reconstruct the shortest path from the start node to the end node and return this path 
    along with the time taken to traverse it and the number of nodes visited during the process.

    """
    edges = {}
    max_speed = 0
    with open(edgeFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            s = int(row['start'])
            e = int(row['end'])
            d = float(row['distance'])
            t = d / (float(row['speed limit']) / 3.6)
            if s in edges:
                edges[s] += [(e, d, t)]
            else:
                edges[s] = [(e, d, t)]
            if float(row['speed limit']) > max_speed:
                max_speed = float(row['speed limit'])
    heuristic = {}
    with open(heuristicFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            heuristic[int(row['node'])] = float(row[str(end)]) / (max_speed / 3.6)
    times = {}
    Q = []
    heapq.heappush(Q, (0 + heuristic[start], start, None))
    while len(Q) > 0:
        cur_time, cur_node, prev_node = heapq.heappop(Q)
        if cur_node in times:
            continue
        times[cur_node] = (cur_time, prev_node)
        if cur_node == end:
            break
        if not cur_node in edges:
            continue
        for nxt_node, d, t in edges[cur_node]:
            if nxt_node in times:
                continue
            heapq.heappush(Q, (cur_time + t - heuristic[cur_node] + heuristic[nxt_node], nxt_node, cur_node))
    path = [end]
    while times[path[-1]][1] != None:
        path.append(times[path[-1]][1])
    path.reverse()
    time = times[end][0]
    num_visited = len(times)
    return path, time, num_visited
    #raise NotImplementedError("To be implemented")
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
