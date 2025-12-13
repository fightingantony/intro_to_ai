import csv
from queue import PriorityQueue
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
    """
    Implementing UCS algorithm, I read the graph data from a CSV file using csv.DictReader, storing it 
    as an adjacency list in the edge dictionary. Then I initialize a dictionary ‘distance’ to track the 
    shortest distance from the start node to each node, and a priority queue ‘pq’ to store nodes based 
    on their priority (current distance from the start node). Next, repeatedly dequeue nodes from the 
    priority queue until it finds the target node or the priority queue becomes empty. For each 
    dequeued node, we explore its adjacent nodes. If a shorter path to an adjacent node is found,
    update its distance in the dictionary ‘distance’ and enqueue the adjacent node along with its 
    updated distance into the priority queue. Once the target node is reached, reconstruct the 
    shortest path from the end node to the start node using pointers stored in the distance dictionary. 
    and return the reconstructed path in reverse order, along with the distance of the shortest path 
    and the number of visited nodes.
    """
    edge = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            s = int(row['start'])
            e = int(row['end'])
            d = float(row['distance'])
            if s in edge:
                edge[s] += [(e, d)]
            else:
                edge[s] = [(e, d)]
    distance = {start: (0, None)}
    pq = PriorityQueue()
    pq.put((0, start))
    while not pq.empty():
        cur_dist, cur_node = pq.get()
        if cur_node == end:
            break
        if cur_node not in edge:
            continue
        for next_node, next_dist in edge[cur_node]:
            new_dist = cur_dist + next_dist
            if next_node not in distance or new_dist < distance[next_node][0]:
                distance[next_node] = (new_dist, cur_node)
                pq.put((new_dist, next_node))
    path = [end]
    while path[-1] != start:
        path.append(distance[path[-1]][1])
    path.reverse()
    dist = distance[end][0]
    num_visited = len(distance)
    return path, dist, num_visited
    #raise NotImplementedError("To be implemented")
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
