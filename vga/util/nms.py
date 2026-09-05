def nms(boxes, scores, iou_threshold):
    n = len(boxes)
    if n == 0:
        return []
    
    def iou(box1, box2):
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        
        x1_inter = max(x1a, x1b)
        y1_inter = max(y1a, y1b)
        x2_inter = min(x2a, x2b)
        y2_inter = min(y2a, y2b)
        
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        area1 = (x2a - x1a) * (y2a - y1a)
        area2 = (x2b - x1b) * (y2b - y1b)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    
    indices = sorted(range(n), key=lambda i: (-scores[i], i))
    removed = set()
    kept = []
    
    for i in indices:
        if i in removed:
            continue
        kept.append(i)
        for j in range(n):
            if j == i or j in removed:
                continue
            if iou(boxes[i], boxes[j]) > iou_threshold:
                removed.add(j)
    
    return kept
