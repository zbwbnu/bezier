import taichi as ti
import numpy as np

# Taichi 初始化
ti.init(arch=ti.gpu)

# 窗口尺寸
W = 800
H = 800

# 最大控制点 & 曲线精度
MAX_PTS = 100
SAMPLING = 1000

# 像素场
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

# 控制点存储
control_pt = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PTS)
line_indices = ti.field(dtype=ti.i32, shape=MAX_PTS * 2)

# 曲线点缓存
curve_pt = ti.Vector.field(2, dtype=ti.f32, shape=SAMPLING)

# 组合数：C(n, k)
def comb(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

# 伯恩斯坦多项式计算贝塞尔点
def bezier_pos(points, t):
    n = len(points) - 1
    x = 0.0
    y = 0.0
    for i in range(n + 1):
        c = comb(n, i)
        term = c * (t ** i) * ((1 - t) ** (n - i))
        x += points[i][0] * term
        y += points[i][1] * term
    return [x, y]

# 清空屏幕
@ti.kernel
def clear_screen():
    for i, j in color_buffer:
        color_buffer[i, j] = [0.0, 0.0, 0.0]

# GPU 绘制曲线
@ti.kernel
def render_curve(total: ti.i32):
    for idx in range(total):
        p = curve_pt[idx]
        px = ti.cast(p[0] * W, ti.i32)
        py = ti.cast(p[1] * H, ti.i32)
        if 0 <= px < W and 0 <= py < H:
            color_buffer[px, py] = [0.0, 0.9, 0.3]

# 主程序
def main():
    win = ti.ui.Window("Bezier Curve (New Version)", (W, H))
    canvas = win.get_canvas()
    user_points = []

    while win.running:
        # 事件监听
        for evt in win.get_events(ti.ui.PRESS):
            if evt.key == ti.ui.LMB:
                if len(user_points) < MAX_PTS:
                    cursor = win.get_cursor_pos()
                    user_points.append(cursor)
            elif evt.key == 'c':
                user_points.clear()

        clear_screen()
        count = len(user_points)

        # 计算贝塞尔曲线
        if count >= 2:
            curve_np = np.zeros((SAMPLING, 2), dtype=np.float32)
            for step in range(SAMPLING):
                t = step / (SAMPLING - 1)
                curve_np[step] = bezier_pos(user_points, t)
            curve_pt.from_numpy(curve_np)
            render_curve(SAMPLING)

        # 显示画面
        canvas.set_image(color_buffer)

        # 绘制控制点
        if count > 0:
            np_p = np.full((MAX_PTS, 2), -10.0, dtype=np.float32)
            np_p[:count] = user_points
            control_pt.from_numpy(np_p)
            canvas.circles(control_pt, radius=0.007, color=(1, 0, 0))

            # 绘制控制线
            if count >= 2:
                idx_list = []
                for i in range(count - 1):
                    idx_list += [i, i + 1]
                np_idx = np.zeros(MAX_PTS * 2, dtype=np.int32)
                np_idx[:len(idx_list)] = idx_list
                line_indices.from_numpy(np_idx)
                canvas.lines(control_pt, indices=line_indices, width=0.002, color=(0.5, 0.5, 0.5))

        win.show()

if __name__ == "__main__":
    main()