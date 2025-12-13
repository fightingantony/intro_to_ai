import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    """
    Same as BFS, I read the graph data from a CSV file using csv.DictReader, storing it as an adjacency 
    list in the edge dictionary. Then I implement DFS using a stack, I initialize a stack with the start
    node and its distance 0, and a dictionary ‘distance’ to track the shortest distance from the start 
    node to each node. Next, I start DFS traversal from the start node. It iteratively pops nodes from 
    the stack, explores their adjacent nodes, updates distances, and pushes unvisited adjacent nodes 
    onto the stack until the end node is reached or the stack becomes empty. After that, I reconstruct
    the path from the end node to the start node using pointers stored in the distance dictionary then 
    reverse it, and return the reconstructed path, the distance of the path, and the number of visited 
    nodes.
    """
    edge = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            s = int(row['start'])
            e = int(row['end'])
            d = float(row['distance'])
            if s in edge:
                edge[s].append((e, d))
            else:
                edge[s] = [(e, d)]
    stack = [(start, 0)]
    distance = {start: (0, None)}
    while stack:
        cur_node, cur_dist = stack.pop()
        if cur_node not in edge:
            continue
        for next_node, d in edge[cur_node]:
            if next_node in distance:
                continue
            distance[next_node] = (cur_dist + d, cur_node)
            stack.append((next_node, cur_dist + d))
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
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
