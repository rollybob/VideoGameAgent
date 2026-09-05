def is_stuck(positions, window, tol):
    if len(positions) < window:
        return False
    recent = positions[-window:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    return spread_x <= tol and spread_y <= tol

if __name__ == "__main__":
    pass
