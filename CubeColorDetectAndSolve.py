import cv2
import numpy as np
import struct
from kociemba import solve
from magiccube.cube import Cube
from itertools import product
import json
import os
import threading
from flask import Flask, Response, render_template_string
from io import BytesIO
import time

# ===================== 全局配置 =====================
# 旋转方法定义
ROTATION_METHODS = {"U":["","U","U2","U'"],"D":["","D","D2","D'"],"L":["","L","L2","L'"],
       "R":["","R","R2","R'"],"F":["","F","F2","F'"],"B":["","B","B2","B'"]}

# 颜色定义（W白 Y黄 R红 O橙 G绿 B蓝）
DEFAULT_COLOR_THRESH = {
    "W": [0, 179, 5, 40, 190, 235],
    "Y": [18, 45, 80, 140, 195, 255],
    "R": [0, 8, 135, 200, 186, 250],
    "O": [9, 15, 110, 195, 201, 255],
    "G": [41, 79, 37, 137, 127, 210],
    "B": [73, 135, 136, 235, 166, 235]
}
# 颜色优先级
COLOR_PRIORITY = ["R", "O", "Y", "G", "B", "W"]

# 标准魔方配色：U黄 D白 F绿 B蓝 L红 R橙
CENTER_COLOR_TO_FACE = {
    "Y": "U",
    "W": "D",
    "G": "F",
    "B": "B",
    "R": "L",
    "O": "R"
}
FACE_TO_CENTER_COLOR = {v: k for k, v in CENTER_COLOR_TO_FACE.items()}

ColorTransToFace = str.maketrans(CENTER_COLOR_TO_FACE)
FaceTransToColor = str.maketrans(FACE_TO_CENTER_COLOR)

COLOR_LIST = ["W", "Y", "R", "O", "G", "B"]
FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
COLOR_RGB = {
    "W": (255,255,255), "Y": (0,255,255),   # OpenCV是BGR！
    "R": (0,0,255), "O": (0,165,255),
    "G": (0,255,0), "B": (255,0,0)
}
CALIB_FILE = "cube_calib.json"

# 全局魔方数据
cubeColor = ""
cubeStr = ""
solveStr = ""
cube_data = {face: None for face in FACE_ORDER}

# 摄像头全局变量
cap = None
calib_data = DEFAULT_COLOR_THRESH
roi_box = (200, 100, 200, 300)
roi_locked = True
collecting_finished = False

# 实际的按键 event 号
EVENT_DEVICE = "/dev/input/event1"
EVENT_FORMAT = 'llHHI'
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
KEY_PRESS = 1
KEY_RELEASE = 0
KEY_HOLD = 2

keyValue = 0
show_img = None
title=""
text=""

# ===================== Flask应用 =====================
app_camera = Flask(__name__)
app_plot = Flask(__name__)

CAMERA_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>魔方识别</title>
    <style>body{text-align:center;background:#1a1a1a;color:white;}
    img{max-width:90%;margin-top:20px;border:2px solid #00ff00;}</style>
</head>
<body>
    <h1>魔方实时识别</h1>
    <img src="/video_feed" alt="摄像头流">
</body>
</html>
"""

PLOT_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>魔方展开图</title>
    <style>body{text-align:center;background:#1a1a1a;color:white;}
    img{max-width:95%;margin-top:10px;}</style>
</head>
<body>
    <h1>魔方实时展开图</h1>
    <img src="/plot_feed" alt="魔方绘图">
</body>
</html>
"""

# ===================== 旋转函数 =====================
def rotate_face(face, turns):
    f = list(face)
    for _ in range(turns % 4):
        f = [f[6], f[3], f[0], f[7], f[4], f[1], f[8], f[5], f[2]]
    return f

def build_cube(cube_str, rotations):
    faces = [cube_str[9*i:9*i+9] for i in range(6)]
    rotated = [rotate_face(f, r) for f, r in zip(faces, rotations)]
    return ''.join(''.join(f) for f in rotated)

def find_valid_cube_and_solve():
    global cubeColor, cubeStr, solveStr, cube_data,title,text
    for rot in product([0,1,2,3], repeat=6):
        cubeStrTmp = build_cube(cubeStr, rot)
        if solve_cube(cubeStrTmp):
            cubeStr = cubeStrTmp
            cubeColor = cubeStr.translate(FaceTransToColor)
            for i, f in enumerate(FACE_ORDER):
                cube_data[f] = [list(cubeColor[i*9+3*j:i*9+3*j+3]) for j in range(3)]
            print("魔方输入状态:", cubeStr)
            print(f"还原步骤：{solveStr}")
            title = "Cube Solve - waiting for key press"
            text = f"Full Steps:{solveStr}"
            return True
    print("非法魔方状态无法求解")
    title = "Cube Detect - Invalid Cube!"
    text = "Press key to reset and collect again"
    return False

# ===================== 颜色识别 =====================
def get_color_by_thresh(hsv_roi, calib):
# 去噪预处理
    hsv_roi = cv2.GaussianBlur(hsv_roi, (5, 5), 0)
    
    color_pixel = {}
    min_valid_pixel = 20  # 过滤噪点
    
    for color in COLOR_PRIORITY:
        if color not in calib:
            continue
        
        h_min, h_max, s_min, s_max, v_min, v_max = calib[color]
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv_roi, lower, upper)
        
        # 形态学去噪
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        count = cv2.countNonZero(mask)
        if count >= min_valid_pixel:
            color_pixel[color] = count

    if not color_pixel:
        return "W"
    
    best_color = max(color_pixel, key=color_pixel.get)
    return best_color

# ===================== 魔方旋转 =====================
def cube_rotation(rotationMethod):
    global cubeColor,cubeStr,cube_data
    cubeColorTmp = cubeColor[:9] + cubeColor[36:45] + cubeColor[18:27]+ \
                   cubeColor[9:18] + cubeColor[45:] + cubeColor[27:36]
    c = Cube(3, cubeColorTmp)
    c.rotate(rotationMethod)
    
    lines = [line.strip() for line in str(c).splitlines()]
    U = ''.join(lines[i].replace(' ','') for i in range(3))
    mid = lines[3:6]
    B = ''.join(row[:9].replace(' ','') for row in mid)
    F = ''.join(row[9:18].replace(' ','') for row in mid)
    L = ''.join(row[18:27].replace(' ','') for row in mid)
    D = ''.join(row[27:].replace(' ','') for row in mid)
    R = ''.join(lines[i].replace(' ','') for i in range(6,9))

    cubeColor = U + L + F + R + B + D
    cubeStr = cubeColor.translate(ColorTransToFace)

    for i, f in enumerate(FACE_ORDER):
        cube_data[f] = [list(cubeColor[i*9+3*j:i*9+3*j+3]) for j in range(3)]

# ===================== OpenCV 绘制魔方展开图（核心！）=====================
# 展开图布局参数
CELL_SIZE = 40
GAP = 5
FACE_POS = {
    "U": (6*CELL_SIZE, 3*CELL_SIZE),
    "L": (3*CELL_SIZE, 6*CELL_SIZE),
    "F": (6*CELL_SIZE, 6*CELL_SIZE),
    "R": (9*CELL_SIZE, 6*CELL_SIZE),
    "B": (12*CELL_SIZE, 6*CELL_SIZE),
    "D": (6*CELL_SIZE, 9*CELL_SIZE)
}

# 绘图函数（纯OpenCV）
def draw_cube_cv2():
    global cube_data, title, text

    # 创建黑色背景图
    img_height =12 * CELL_SIZE
    img_width = 18 * CELL_SIZE
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img[:] = (30,30,30)  # 深灰背景

    # if os.path.exists(CALIB_FILE):
    #     with open(CALIB_FILE) as f:
    #         calibTmp = json.load(f)
    #         if(calib_data != calibTmp):
    #             calib_data = calibTmp
    #             print("颜色标定数据已更新")
            
    # 绘制每个面
    for face, (x0, y0) in FACE_POS.items():
        mat = cube_data[face]

        # 绘制3x3格子
        for row in range(3):
            for col in range(3):
                x = x0 + col * CELL_SIZE
                y = y0 + row * CELL_SIZE

                # 颜色
                if mat is None:
                    color = (100,100,100)
                else:
                    color_key = mat[row][col]
                    color = COLOR_RGB[color_key]

                # 画色块
                cv2.rectangle(img, (x+1, y+1), (x+CELL_SIZE-1, y+CELL_SIZE-1), color, -1)
                # 画黑边框
                cv2.rectangle(img, (x, y), (x+CELL_SIZE, y+CELL_SIZE), (0,0,0), 1)

        # 面中心文字
        cx = x0 + CELL_SIZE*1 + CELL_SIZE//2
        cy = y0 + CELL_SIZE*1 + CELL_SIZE//2
        cv2.putText(img, face, (cx-8, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # 标题
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(img, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return img

# ===================== 推流生成器 =====================
def gen_camera():
    global cap, collecting_finished, show_img
    while True:
        if collecting_finished == True:
            success, show_img = cap.read()
            if not success: break
        cv2.putText(show_img, "Cube Detect", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        ret, jpeg = cv2.imencode('.jpg', show_img)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def gen_plot():
    while True:
        img = draw_cube_cv2()
        ret, jpeg = cv2.imencode('.jpg', img)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ===================== 按键读取 =====================
def read_key_event():
    global keyValue
    with open(EVENT_DEVICE, "rb") as f:
        print(f"正在监听按键: {EVENT_DEVICE}")
        while True:
            event = f.read(EVENT_SIZE)
            if not event: continue
            tv_sec, tv_usec, type, code, value = struct.unpack(EVENT_FORMAT, event)
            if type == 1:
                if value == KEY_PRESS:
                    keyValue = 1
                    print("按键按下")

# ===================== Flask路由 =====================
@app_camera.route('/')
def index_camera():
    return render_template_string(CAMERA_PAGE)

@app_camera.route('/video_feed')
def video_feed():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app_plot.route('/')
def index_plot():
    return render_template_string(PLOT_PAGE)

@app_plot.route('/plot_feed')
def plot_feed():
    return Response(gen_plot(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===================== 魔方求解 =====================
def solve_cube(cube_face):
    global solveStr
    try:
        if cube_face == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB":
            solveStr = ""
            return True
        solveStr = solve(cube_face)
        return True
    except:
        return False

def visualize_solve_background():
    global solveStr, title, text
    if not solveStr: return
    steps = solveStr.split()
    for i, step in enumerate(steps):
        title = f"Cube Solve - Solving Step {i+1}/{len(steps)}" + ": " + step
        text = f"Full Steps:{solveStr}"
        cube_rotation(step)
        time.sleep(0.5)
    title = "Cube Solved: " + solveStr
    text = "Waiting for key press to reset"
    return True

# ===================== 自动采集魔方 =====================
def auto_collect_cube():
    global calib_data,cubeColor, cubeStr, solveStr, cube_data, collecting_finished, keyValue, show_img, cap, title, text
   
    while True:
        #测试模式
        if 1:
            collected = set()
            collecting_finished = False
            cubeColor = ""
            cubeStr = ""
            solveStr = ""
            cube_data = {face: None for face in FACE_ORDER}

            while len(collected) < 6:
                title = f"Cube Detect - Collecting face {len(collected)+1}/6"
                text = "Press key to start collecting cube colors"
                ret, frame = cap.read()
                if not ret: break
                show_img = frame.copy()
                x, y, w, h = roi_box
                face_img = frame[y:y+h, x:x+w]
                hsv_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
                cell_w, cell_h = w//3, h//3

                mat = []
                for row in range(3):
                    r = []
                    for col in range(3):
                        cx1 = col*cell_w + cell_w//5
                        cy1 = row*cell_h + cell_h//5
                        cx2 = (col+1)*cell_w - cell_w//5
                        cy2 = (row+1)*cell_h - cell_h//5
                        cell_roi = hsv_face[cy1:cy2, cx1:cx2]
                        color = get_color_by_thresh(cell_roi, calib_data)
                        cv2.rectangle(face_img, (col*cell_w, row*cell_h), 
                                    ((col+1)*cell_w, (row+1)*cell_h), (0,255,0), 2)
                        cv2.putText(face_img, color, (col*cell_w+8, row*cell_h+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        r.append(color)
                    mat.append(r)
                show_img[y:y+h, x:x+w] = face_img
                
                center = mat[1][1]
                if center in CENTER_COLOR_TO_FACE and keyValue == 1:
                    keyValue = 0
                    face = CENTER_COLOR_TO_FACE[center]
                    cube_data[face] = mat
                    collected.add(center)
                    print(f"已采集: {len(collected)}/6")
            collecting_finished = True

            res = []
            for f in FACE_ORDER:
                mat = cube_data[f]
                for row in mat:
                    res.extend(row)
            cubeColor = ''.join(res)
            cubeStr = cubeColor.translate(ColorTransToFace)
        else:
            collecting_finished = True
            cubeStr = "RRLDURBUFLBDFRLFFBUBURFURBDBLRBDDFDDURLFLFRDDFUBUBLLLU"
            cubeColor = cubeStr.translate(FaceTransToColor)
            for i, f in enumerate(FACE_ORDER):
                cube_data[f] = [list(cubeColor[i*9+3*j:i*9+3*j+3]) for j in range(3)]

        title = "Cube Detect - Collecting Finished!"
        text = "Press key to find valid cube and solve"
        cube_valid = False
        cube_solve_finished = False
        cube_reset = False

        while cube_reset == False:
            if keyValue == 1 and collecting_finished == True and cube_valid == False:
                keyValue = 0
                cube_valid = find_valid_cube_and_solve()

            if keyValue == 1 and cube_valid == True and cube_solve_finished == False:
                keyValue = 0
                cube_solve_finished = visualize_solve_background()    

            if keyValue == 1 and cube_solve_finished == True:
                keyValue = 0
                cube_reset = True
                break

def calib_event():
    global calib_data
    while True:
        if os.path.exists(CALIB_FILE):
            with open(CALIB_FILE) as f:
                calibTmp = json.load(f)
                if(calib_data != calibTmp):
                    calib_data = calibTmp
                    print("加载颜色标定")
        else:
            calib_data = DEFAULT_COLOR_THRESH
        time.sleep(2)


# ===================== 启动服务 =====================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 启动线程
    threading.Thread(target=calib_event, daemon=True).start()
    threading.Thread(target=auto_collect_cube, daemon=True).start()
    threading.Thread(target=read_key_event, daemon=True).start()
    threading.Thread(target=app_camera.run, kwargs={"host":"0.0.0.0","port":5000,"debug":False}, daemon=True).start()
    threading.Thread(target=app_plot.run, kwargs={"host":"0.0.0.0","port":5001,"debug":False}, daemon=True).start()

    print("\n===== NXP-IMX91魔方识别及还原系统 =====")
    print("摄像头识别页面: http://IP:5000")
    print("魔方绘图页面: http://IP:5001")
    print("=================================\n")

    try:
        while True:
            input()
    except KeyboardInterrupt:
        cap.release()
        print("\n服务已停止")