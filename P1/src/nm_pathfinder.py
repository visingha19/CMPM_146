import math
import heapq

def find_path(source_point, destination_point, mesh):

    """
    Searches for a path from source_point to destination_point through the mesh

    Args:
        source_point: starting point of the pathfinder
        destination_point: the ultimate goal the pathfinder must reach
        mesh: pathway constraints the path adheres to

    Returns:

        A path (list of points) from source_point to destination_point if exists
        A list of boxes explored by the algorithm
    """
    
    start_box = point_to_box(source_point, mesh['boxes'])
    goal_box = point_to_box(destination_point, mesh['boxes'])

    if start_box is None or goal_box is None:
        return [], []

    if start_box == goal_box:
        return [source_point, destination_point], [start_box]

    # Forward and backward, since it is A*
    forwardsOpen = []
    backwardsOpen = []
    
    # Costs associated with movement
    forwardsCost = {start_box: 0.0}
    backwardsCost = {goal_box: 0.0}

    # For path reconstruction: store how path was arrived at    
    f_parent = {start_box: None}
    b_parent = {goal_box: None}

    heapq.heappush(forwardsOpen, (heuristic_box(start_box, goal_box), start_box))
    heapq.heappush(backwardsOpen, (heuristic_box(goal_box, start_box), goal_box))

    visited_boxes = set()

    while forwardsOpen and backwardsOpen:
        
        f_current_cost, f_current = heapq.heappop(forwardsOpen)
        visited_boxes.add(f_current)

        # If f_current is found from back side, there is a meeting point
        if f_current in b_parent:
            return reconstruct_path_bidirectional(
                f_current, f_parent, b_parent,
                source_point, destination_point
            ), visited_boxes

        # Expand from the neighbors
        for neighbor in mesh['adj'].get(f_current, []):
            new_cost = forwardsCost[f_current] + cost_between_boxes(f_current, neighbor)
            if neighbor not in forwardsCost or new_cost < forwardsCost[neighbor]:
                forwardsCost[neighbor] = new_cost
                f_parent[neighbor] = f_current
                priority = new_cost + heuristic_box(neighbor, goal_box)
                heapq.heappush(forwardsOpen, (priority, neighbor))

        # Expand from the back side
        b_current_cost, b_current = heapq.heappop(backwardsOpen)
        visited_boxes.add(b_current)

        # If b_current is found from forward side, there is a meeting point
        if b_current in f_parent:
            return reconstruct_path_bidirectional(
                b_current, f_parent, b_parent,
                source_point, destination_point
            ), visited_boxes

        # Expand from the neighbors
        for neighbor in mesh['adj'].get(b_current, []):
            new_cost = backwardsCost[b_current] + cost_between_boxes(b_current, neighbor)
            if neighbor not in backwardsCost or new_cost < backwardsCost[neighbor]:
                backwardsCost[neighbor] = new_cost
                b_parent[neighbor] = b_current
                priority = new_cost + heuristic_box(neighbor, start_box)
                heapq.heappush(backwardsOpen, (priority, neighbor))

    # No path found
    return [], visited_boxes


def point_to_box(point, boxes):
    # Given point and list of boxes, return box containing the point
    x, y = point
    for box in boxes:
        (x1, x2, y1, y2) = box
        # Determine whether point is within bounds of the box
        if x1 <= x < x2 and y1 <= y < y2:
            return box
    return None


def center_of_box(box):
    # Return center of a box (helper function)
    x1, x2, y1, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def heuristic_box(a, b):
    # Heuristic for A* used to find the distance between centers of box a and box b
    center_a = center_of_box(a)
    center_b = center_of_box(b)
    return math.dist(center_a, center_b)


def cost_between_boxes(a, b):
    # Cost of moving between adjacent boxes a and b, using distance between centers
    return heuristic_box(a, b)


def reconstruct_path_bidirectional(meeting_box, f_parent, b_parent, source_point, destination_point):
    # Reconstruct path when the meeting box is discovered from both sides

    # Step 1: Build from meeting_box back to start_box using f_parent
    forward_path = []
    cur = meeting_box
    while cur is not None:
        forward_path.append(cur)
        cur = f_parent[cur]
    forward_path.reverse()

    # Step 2: Build from meeting_box back to goal_box using b_parent
    backward_path = []
    cur = meeting_box
    while cur is not None:
        backward_path.append(cur)
        cur = b_parent[cur]

    # Step 3: Reverse second part and combine them
    combined_boxes = forward_path + backward_path[1:]

    # Step 4: Place source and destination point at the end
    path = [source_point]
    for box in combined_boxes:
        path.append(center_of_box(box))
    path.append(destination_point)

    return path

if __name__ == "__main__":
    # Testing
    test_mesh = {
        'boxes': [
            # box1: 
            (0, 10, 0, 10),
            # box2:
            (0, 10, 10, 20),
        ],
        'adj': {
            (0, 10, 0, 10): [(0, 10, 10, 20)],
            (0, 10, 10, 20): [(0, 10, 0, 10)],
        }
    }

    source = (5, 5)
    dest   = (5, 15)

    path, visited = find_path(source, dest, test_mesh)

    print("Found path:", path)
    print("Visited boxes:", visited)
