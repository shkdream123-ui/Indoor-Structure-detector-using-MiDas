# Python 3.12
import socket
import struct
import numpy as np
import cv2
from collections import deque
import threading
import math
import time
import random

HOST = '0.0.0.0'
PORT = 5000

EXPECTED_W = 256
EXPECTED_H = 256

# ------------------- 큐 -------------------
raw_queue = deque(maxlen=1)
processed_queue = deque(maxlen=1)
gyro_queue = deque(maxlen=1000)# 추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.
accel_queue = deque(maxlen=1000)# 추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.

# ------------------- Depth 노멀라이즈 -------------------
def normalize_depth(depth_frame):
    depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(depth_frame)
    max_val = np.max(depth_frame)
    depth_norm = (depth_frame - min_val) / (max_val - min_val + 1e-6)
    return (depth_norm * 255).astype(np.uint8)

# ------------------- 가중치 계산 -------------------
def compute_depth_confidence(depth_small):
    depth_f = depth_small.astype(np.float32)

    # 1. gradient 계산
    grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # 2. 블록 단위로 downsample → 공간적 분포 파악 목적
    # 16x16 정도면 충분 (전체가 256x256이라면 16배 축소)
    block = cv2.resize(grad_mag, (16, 16), interpolation=cv2.INTER_AREA)

    # 3. 블록들의 분산 계산
    spatial_var = block.var()  # 하나의 수

    # 4. sigmoid로 0~1 스케일로 압축 (필요하면)
    # C 값은 조정 가능. variance가 0.0~0.01 정도 나온다면 C=0.005 정도
    C = 0.005
    confidence = 1.0 - math.exp(-spatial_var / C)

    return float(np.clip(confidence, 0.0, 1.0))


def compute_flow_weight(flow_mag_map, k=10.0, x0=0.5):
    """
    flow_mag_map : optical flow magnitude map
    k : sigmoid 기울기, 작게 하면 완만
    x0 : sigmoid 중앙값, W=0.5가 되는 std 기준
    """
    std = float(flow_mag_map.std())
    W = 1.0 / (1.0 + np.exp(-k * (std - x0)))
    return np.clip(W, 0.0, 1.0)

def compute_static_weight(gray_frame, NX, NY):
    gray_umat = cv2.UMat(gray_frame)
    small = cv2.resize(gray_umat, (NX//4, NY//4), interpolation=cv2.INTER_AREA).get()
    std = float(small.std())
    C = 50.0
    W = 1.0 - math.exp(-std / C)
    return np.clip(W, 0.0, 1.0)

#-------------------- 필터링 행렬 -----------------
def compute_depth_matrix(depth_uint8, threshold=0.7, k=20.0):
    # 1) 0~1 정규화
    depth_norm = depth_uint8.astype(np.float32) / 255.0

    # 2) 시그모이드 함수 적용
    #    threshold에서 0.5가 되도록 중심을 이동
    #    k는 기울기(가파름)
    W_depth = 1.0 / (1.0 + np.exp(-k * (depth_norm - threshold)))

    return W_depth

def compute_flow_matrix(flow_norm, f_wflow, k=20.0):
    """
    flow_norm: 이미 0~1로 정규화된 optical flow magnitude 값
    f_wflow : 유효값(= sigmoid의 중앙값, 출력이 0.5가 되는 지점)
    k : 기울기 조절 상수 (20~30 권장)
    """
    flow = np.clip(flow_norm - f_wflow,0.0,1.0)
    
    W_flow = 1.0 / (1.0 + np.exp(-k * (flow-0.5)))
    
    return W_flow

# ------------------edge 연결 함수 ----------------
def fill_edges(mask):
    # mask: numpy uint8 binary (0/255)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)
    return mask_filled

# ------------------- TCP 수신 -------------------
def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_thread(sock, raw_queue, gyro_queue, accel_queue):
    """
    Packet format
    ------------------------------------------------
    [1B] packet_type
      0x01 : frame packet
        [4B] jpeg_len
        [4B] depth_len
        [jpeg bytes]
        [depth bytes]

      0x02 : gyro packet
        [8B] timestamp (ns)
        [4B] gyro_z (rad/s)

      0x03 : accel packet
        [8B] timestamp (ns)
        [4B] accel_x (m/s^2)
        [4B] accel_y (m/s^2)
        [4B] accel_z (m/s^2)

    ------------------------------------------------
    """

    while True:
        # -----------------------------
        # 1) packet type
        # -----------------------------
        pkt_type_raw = recv_all(sock, 1)
        if not pkt_type_raw:
            print("[RECV] connection closed")
            break

        pkt_type = pkt_type_raw[0]

        # -----------------------------
        # 2) FRAME PACKET
        # -----------------------------
        if pkt_type == 0x01:
            header = recv_all(sock, 8)
            if not header:
                break

            jpeg_len, depth_len = struct.unpack('!II', header)

            jpeg_bytes = recv_all(sock, jpeg_len)
            depth_bytes = recv_all(sock, depth_len)

            if jpeg_bytes is None or depth_bytes is None:
                print("[RECV] frame incomplete")
                break

            raw_queue.append(
                (jpeg_bytes, depth_bytes)
            )

        # -----------------------------
        # 3) GYRO PACKET  (추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.)
        # -----------------------------
        elif pkt_type == 0x02:
            payload = recv_all(sock, 12)
            if not payload:
                break

            timestamp, gyro_z = struct.unpack('!qf', payload)

            #print(f"Received gyro: timestamp={timestamp}, gyroZ={gyro_z}")

            gyro_queue.append(
                (timestamp, gyro_z)
            )

        # ------------------------------
        # 4) ACCEL PACKET
        # ------------------------------
        elif pkt_type == 0x03:
            payload = recv_all(sock, 20)
            if not payload:
                break

            timestamp, ax, ay, az = struct.unpack('!qfff', payload)

            pc_ts = time.monotonic_ns()

            #print(f"Received accel: timestamp={timestamp}, ax={ax}, ay={ay}, az={az}")

            accel_queue.append(
                (timestamp, pc_ts, ax, ay, az)
            )


        else:
            print(f"[RECV] unknown packet type: {pkt_type}")
            break
        

# ------------------ 유틸 ------------------

def line_to_abc(r, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    c = -r
    return a, b, c

def is_inside(vp, w, h):
    return (0 <= vp[0] <= w) and (0 <= vp[1] <= h)

def intersect(line1, line2):
    r1, theta1 = line1
    r2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([r1, r2])

    try:
        x, y = np.linalg.solve(A, b)
        return (x, y)
    except np.linalg.LinAlgError:
        return None

def compute_angle_entropy(lines, num_bins=36):
    if lines is None or len(lines) == 0:
        return 0.0

    angles = np.array([theta for _, theta in lines])
    angles = angles % np.pi

    hist, _ = np.histogram(angles, bins=num_bins, range=(0, np.pi), density=True)

    hist = hist[hist > 1e-6]

    if len(hist) == 0:
        return 0.0

    entropy = -np.sum(hist * np.log(hist))

    return float(entropy)   # 🔥 확실히 float 반환

def classify_angle(theta):
    theta = float(theta)
    theta = theta % np.pi

    # 수직 (90도)
    if abs(theta - np.pi/2) < 15 * np.pi/180:
        return "VERTICAL"

    # 수평 (0도)
    elif abs(theta - 0) < 15 * np.pi/180 or abs(theta - np.pi) < 10 * np.pi/180:
        return "HORIZONTAL"

    else:
        return "OBLIQUE"

def fill_depth_holes(depth, hole_mask, angle_grad, depth_grad, num_iters=5):
    """
    depth: normalized depth (H, W)
    hole_mask: True = 채워야 할 영역
    angle_grad: gradient 방향
    depth_grad: gradient magnitude
    """

    filled = depth.copy()

    H, W = depth.shape

    for _ in range(num_iters):
        new_filled = filled.copy()

        ys, xs = np.where(hole_mask)

        for y, x in zip(ys, xs):

            vals = []
            weights = []

            # 8방향 탐색
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:

                    if dx == 0 and dy == 0:
                        continue

                    ny, nx = y + dy, x + dx

                    if 0 <= ny < H and 0 <= nx < W:
                        if not hole_mask[ny, nx]:

                            # 방향 정렬 weight
                            dir_vec = np.array([dx, dy], dtype=np.float32)
                            dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)

                            grad_angle = angle_grad[ny, nx]
                            grad_vec = np.array([
                                np.cos(grad_angle),
                                np.sin(grad_angle)
                            ])

                            align = abs(np.dot(dir_vec, grad_vec))

                            w = 0.5 + 0.5 * align  # 방향 일치하면 weight ↑

                            # gradient magnitude 반영
                            w *= (1.0 + depth_grad[ny, nx])

                            vals.append(filled[ny, nx])
                            weights.append(w)

            if len(vals) > 0:
                vals = np.array(vals)
                weights = np.array(weights)

                new_filled[y, x] = np.sum(vals * weights) / (np.sum(weights) + 1e-6)

        filled = new_filled

    return filled

def ransac_line_weighted(points, mode="WALL", iterations=100):

    best_line = None
    best_score = -1

    for _ in range(iterations):

        p1, p2 = random.sample(points, 2)

        # line 생성
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = -(a*p1[0] + b*p1[1])

        norm = np.sqrt(a*a + b*b) + 1e-6
        a, b, c = a/norm, b/norm, c/norm

        # inliers
        inliers = []
        for x, z in points:
            dist = abs(a*x + b*z + c)
            if dist < 0.02:
                inliers.append((x, z))

        if len(inliers) < 10:
            continue

        # -------------------------
        # 🔥 방향
        dx = b
        dz = -a
        angle = abs(np.arctan2(dz, dx))

        # -------------------------
        # 🔥 span
        xs = [p[0] for p in inliers]
        zs = [p[1] for p in inliers]

        x_span = max(xs) - min(xs)
        z_span = max(zs) - min(zs)

        # -------------------------
        # 🔥 중심 편향
        center_bias = abs(np.mean(xs))

        # -------------------------
        # 🔥 angle score
        if mode == "WALL":
            angle_score = wall_angle_score(angle)
        else:
            angle_score = corner_angle_score(angle)

        # -------------------------
        # 🔥 최종 score (핵심🔥)
        score = (
            len(inliers) * 0.5 +
            x_span * 50.0 +          # 🔥 가장 중요
            z_span * 20.0 +          # 보조
            angle_score * 30.0 -     # 방향
            center_bias * 10.0       # 중앙 몰림 패널티
        )

        if score > best_score:
            best_score = score
            best_line = (a, b, c)
            best_inliers = inliers

    return best_line, best_inliers

def extract_two_directions(grad_dirs):

    angles = []

    for dx, dz in grad_dirs:
        angle = np.arctan2(dz, dx)
        angles.append(angle)

    angles = np.array(angles)

    # 🔥 히스토그램
    num_bins = 36
    hist, bin_edges = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))

    top_bins = np.argsort(hist)[-2:]

    dir_list = []

    for b in top_bins:
        mask = (angles >= bin_edges[b]) & (angles < bin_edges[b+1])

        if np.sum(mask) < 5:
            continue

        dx_mean = np.mean([grad_dirs[i][0] for i in range(len(grad_dirs)) if mask[i]])
        dz_mean = np.mean([grad_dirs[i][1] for i in range(len(grad_dirs)) if mask[i]])

        norm = np.sqrt(dx_mean**2 + dz_mean**2) + 1e-6
        dir_list.append((dx_mean/norm, dz_mean/norm))

    return dir_list

def ransac_direction(grad_dirs, num_iters=100):

    best_dir = None
    best_score = 0

    for _ in range(num_iters):
        i = np.random.randint(len(grad_dirs))
        dx, dz = grad_dirs[i]

        score = 0
        for dx2, dz2 in grad_dirs:
            cos_sim = abs(dx*dx2 + dz*dz2)
            score += cos_sim

        if score > best_score:
            best_score = score
            best_dir = (dx, dz)

    return best_dir

def direction_to_line(direction, center):

    dx, dz = direction

    # 법선 벡터 (수직)
    a = -dz
    b = dx

    x0, z0 = center
    c = -(a*x0 + b*z0)

    return (a, b, c)

def pca_line_fit(points):
    pts = np.array(points)

    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)

    direction = eigvecs[:, np.argmax(eigvals)]

    dx, dz = direction

    a = -dz
    b = dx
    c = -(a*mean[0] + b*mean[1])

    norm = np.sqrt(a*a + b*b) + 1e-6
    return (a/norm, b/norm, c/norm)

def project_points(points, line):
    a, b, c = line
    projected = []

    for x, z in points:
        d = (a*x + b*z + c)
        xp = x - a*d
        zp = z - b*d
        projected.append((xp, zp))

    return projected

def project_points_soft(points, line, alpha=0.5):
    a, b, c = line
    projected = []

    for x, z in points:
        d = (a*x + b*z + c)
        xp = x - alpha * a * d
        zp = z - alpha * b * d
        projected.append((xp, zp))

    return projected

def intersect_lines(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2

    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None

    x = (b1*c2 - b2*c1) / det
    z = (c1*a2 - c2*a1) / det

    return (x, z)

def enforce_manhattan_dirs(dir1, dir2):

    dx1, dz1 = dir1

    # dir1 기준으로 수직 방향 생성
    dir2_fixed = (-dz1, dx1)

    return dir1, dir2_fixed

def enforce_manhattan_dirs(dir1, dir2):

    dx1, dz1 = dir1

    # dir1 기준으로 수직 방향 생성
    dir2_fixed = (-dz1, dx1)

    return dir1, dir2_fixed

def enforce_manhattan(line1, line2):
    a1, b1, _ = line1

    # 방향 벡터
    dir1 = np.array([b1, -a1])
    dir1 = dir1 / (np.linalg.norm(dir1) + 1e-6)

    # 수직 방향 생성
    dir2 = np.array([-dir1[1], dir1[0]])

    # line1 유지, line2를 직교로 강제
    a2 = -dir2[1]
    b2 = dir2[0]

    return (line1, (a2, b2, 0))  # c는 나중에 재설정

def wall_angle_score(angle):
    angle = abs(angle)

    max_angle = 40 * np.pi / 180  # 허용 범위

    score = 1.0 - (angle / max_angle)
    return max(score, 0.0)

def wall_angle_score(angle):
    angle = abs(angle)

    max_angle = 20 * np.pi / 180  # 허용 범위

    score = 1.0 - (angle / max_angle)
    return max(score, 0.0)
    
# ------------------- Depth + Saliency 통합 스레드 (최신 frame만) -------------------
def depth_saliency_thread():
    prev_depth = None
    depth_alpha = 0.6
    smoothing_skip = 2
    frame_counter = 0

    depth_vis = True
    edge_vis = True

    while True:
        if not raw_queue:
            time.sleep(0.001)
            continue

        # 항상 최신 frame만
        jpeg_bytes, depth_bytes = raw_queue.pop()
        raw_queue.clear()

        depth_array = np.frombuffer(depth_bytes, dtype='<f4').copy()
        if depth_array.size != EXPECTED_W * EXPECTED_H:
            continue
        depth_frame = depth_array.reshape((EXPECTED_H, EXPECTED_W))

        # Depth downsample + smoothing
        depth_small = cv2.resize(depth_frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame_counter += 1
        if prev_depth is None:
            smoothed = depth_small.copy()
        else:
            if frame_counter % smoothing_skip == 0:
                smoothed = cv2.addWeighted(prev_depth.astype(np.float32),
                                           depth_alpha,
                                           depth_small.astype(np.float32),
                                           1 - depth_alpha, 0.0)
            else:
                smoothed = depth_small.copy()
        prev_depth = smoothed.copy()

        # ---------------- Depth 기반 Object Mask ----------------

        # 1️⃣ Depth 정규화 (기존 유지)
        min_val = np.min(smoothed)
        max_val = np.max(smoothed)
        range_val = max(max_val - min_val, 1e-6)
        depth_norm = (smoothed - min_val) / range_val
        depth_thresh = np.percentile(depth_norm, 75)  # 가까운 25%
        near_mask = depth_norm > depth_thresh

        # ---------------- Depth 기반 edge 생성 ----------------

        # depth gradient
        dzx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        dzy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        angle_grad = np.arctan2(dzy, dzx)
        depth_grad = np.sqrt(dzx**2 + dzy**2)

        # strong depth edge만 선택
        grad_thresh = np.percentile(depth_grad, 90)
        edge_mask = depth_grad > grad_thresh

        # ---------------- ROI 방식 mask 생성 ----------------

        object_candidate = edge_mask & near_mask

        h_d, w_d = depth_norm.shape

        border_ratio = 0.1  # 10% 영역 제거

        y_margin = int(h_d * border_ratio)
        x_margin = int(w_d * border_ratio)

        border_mask = np.zeros_like(depth_norm, dtype=bool)

        border_mask[:y_margin, :] = True
        border_mask[-y_margin:, :] = True
        border_mask[:, :x_margin] = True
        border_mask[:, -x_margin:] = True

        # border 제거
        object_candidate = object_candidate & (~border_mask)

        # dilation (살짝만)
        mask = (object_candidate.astype(np.uint8) * 255)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (5,5), 0)

        # threshold
        _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

        mask = mask.astype(np.uint8)

        # ---------------- 작은 영역 제거 ----------------

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        clean_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:  # ROI보다 조금 크게
                clean_mask[labels == i] = 255

        object_mask = clean_mask > 128
        structure_mask = ~object_mask

        # structure 영역만 선택
        valid = structure_mask & (depth_grad > np.percentile(depth_grad, 50))

        angles_grad = angle_grad[valid]
        weights_grad = depth_grad[valid]

        num_bins_grad = 18
        hist_grad = np.zeros(num_bins_grad)

        for a_g, w_g in zip(angles_grad.flatten(), weights_grad.flatten()):
            bin_idx_g = int((a_g + np.pi) / (2*np.pi) * num_bins_grad)
            bin_idx_g = np.clip(bin_idx_g, 0, num_bins_grad-1)
            hist_grad[bin_idx_g] += w_g

        # top 2 bin index
        top_indices = np.argsort(hist_grad)[::-1][:2]

        b1, b2 = top_indices
        angle1 = b1 * (np.pi / num_bins_grad)
        angle2 = b2 * (np.pi / num_bins_grad)

        def angle_diff(a, b):
            d = abs(a - b)
            return min(d, np.pi - d)

        angle_sep = angle_diff(angle1, angle2)

        # 정규화
        hist_grad = hist_grad / (np.sum(hist_grad) + 1e-6)

        # 상위 peak 찾기
        sorted_hist_grad = np.sort(hist_grad)[::-1]

        peak1_grad = sorted_hist_grad[0]
        peak2_grad = sorted_hist_grad[1]

        # score 정의
        wall_score_grad = peak1_grad
        corner_score_grad = 0.0

        # 각도 차이 기준
        corner_angle_thresh = 20 * np.pi / 180  # 20도

        # 각도 기반 weight
        angle_weight = np.clip((angle_sep - 5*np.pi/180) / (25*np.pi/180), 0, 1)

        # peak2 기반 weight
        peak_weight = np.clip((peak2_grad - 0.02) / 0.2, 0, 1)

        corner_conf = angle_weight * peak_weight

        # score
        corner_score_grad = peak2_grad * (0.5 + angle_weight)
        wall_score_grad = peak1_grad + (1 - corner_conf) * peak2_grad * 0.5

        # ---------------- 6️⃣ 시각화 ----------------

        if depth_vis is True:
            depth_vis_img = (depth_norm * 255).astype(np.uint8)
            grad_vis = (depth_grad / (np.max(depth_grad)+1e-6) * 255).astype(np.uint8)
            obj_vis = (object_mask.astype(np.uint8) * 255)

            depth_resized = cv2.resize(depth_vis_img, (256,256))
            grad_resized = cv2.resize(grad_vis, (256,256))
            obj_resized = cv2.resize(obj_vis, (256,256))

            cv2.imshow("depth", depth_resized)
            cv2.imshow("depth_grad", grad_resized)
            cv2.imshow("object_mask", obj_resized)

        # =================== Saliency ===================
        cam_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        if cam_array is None or cam_array.size == 0:
            print("Empty frame received, skipping...")
            continue
        cam_frame = cv2.imdecode(cam_array, cv2.IMREAD_COLOR)
        if cam_frame is None:
            continue

        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (128,128), interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (3,3), 0)
        small_color = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        edges = cv2.Canny(small, 20, 80)
        h, w = small.shape

        # 🔥 HoughLinesP 사용
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,        # 기존보다 낮춰도 됨
            minLineLength=20,    # 🔥 최소 길이
            maxLineGap=5
        )

        filtered_lines = []

        if lines is not None:
            diag = np.sqrt(h*h + w*w)
            length_threshold = 0.2 * diag

            filtered_lines = []

            # 1️⃣ 길이 필터링
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                if length >= length_threshold:
                    filtered_lines.append((x1, y1, x2, y2, length))

            # 2️⃣ 길이 기준 정렬
            filtered_lines = sorted(filtered_lines, key=lambda l: l[4], reverse=True)

            # 3️⃣ 상위 10% + 최소 5개
            if len(filtered_lines) > 0:
                k = max(5, int(len(filtered_lines) * 0.1))
                dominant_lines = filtered_lines[:k]
            else:
                dominant_lines = []

            # 4️⃣ 시각화
            for x1, y1, x2, y2, _ in dominant_lines:
                cv2.line(small_color, (x1, y1), (x2, y2), (0, 255, 255), 1)

            #print("Filtered:", len(filtered_lines))
            #print("Dominant:", len(dominant_lines))

            # =========================
            # 🔥 5️⃣ 방향 추출
            # =========================

            orientations = []
            weights = []

            for x1, y1, x2, y2, length in dominant_lines:
                theta = np.arctan2(y2 - y1, x2 - x1)

                # 방향 대칭 처리 [0, pi)
                if theta < 0:
                    theta += np.pi

                orientations.append(theta)

                # 🔥 weight = length (or length**2 가능)
                weights.append(length)

            # =========================
            # 🔥 6️⃣ orientation histogram
            # =========================

            num_bins = 18  # 10도 단위
            hist = np.zeros(num_bins)

            for theta, w in zip(orientations, weights):
                bin_idx = int(theta / np.pi * num_bins)
                bin_idx = min(bin_idx, num_bins - 1)
                hist[bin_idx] += w

            # 🔧 optional: smoothing (추천)
            hist = np.convolve(hist, [0.25, 0.5, 0.25], mode='same')

            # =========================
            # 🔥 7️⃣ dominant 방향 추출 (최대 3개) 
            # =========================

            sorted_bins = np.argsort(hist)[::-1]

            top_k = min(3, len(sorted_bins))
            top_bins = sorted_bins[:top_k]

            bin_angle = np.pi / num_bins
            total = np.sum(hist) + 1e-6

            angles = []
            ratios = []

            for b in top_bins:
                angle = b * bin_angle
                ratio = hist[b] / total

                angles.append(angle)
                ratios.append(ratio)

            # =========================
            # 🔥 8️⃣ angle difference 함수
            # =========================

            def angle_diff(a, b):
                d = abs(a - b)
                return min(d, np.pi - d)

            # =========================
            # 🔥 9️⃣ 3번째 방향 검증
            # =========================

            final_angles = []

            if len(angles) >= 1:
                final_angles.append(angles[0])

            if len(angles) >= 2:
                final_angles.append(angles[1])

            # 3번째 방향 검증
            if len(angles) == 3:
                a1, a2, a3 = angles
                r1, r2, r3 = ratios

                d13 = angle_diff(a1, a3)
                d23 = angle_diff(a2, a3)

                # 🔥 기준 (튜닝 가능)
                min_angle_sep = 20 * np.pi / 180   # 최소 20도 이상 떨어져야
                min_ratio = 0.15                   # 충분한 weight 필요

                if (
                    r3 > min_ratio and
                    d13 > min_angle_sep and
                    d23 > min_angle_sep
                ):
                    final_angles.append(a3)  # ✅ 구조 방향으로 인정
                else:
                    pass  # ❌ 노이즈로 제거

            num_directions = len(final_angles)

            # =========================
            # 🔥 final_pairs 생성 (수정)
            # =========================

            final_pairs = []

            for a in final_angles:
                # angles에서 index 찾아서 ratio 매칭
                idx = angles.index(a)
                final_pairs.append((a, ratios[idx]))

            # =========================
            # 🔥 방향 분류
            # =========================

            types = []
            weights_sel = []

            for a, r in final_pairs:
                t = classify_angle(a)
                types.append(t)
                weights_sel.append(r)

            # =========================
            # 🔥 score 계산
            # =========================

            wall_score_edge = 0.0
            corner_score_edge = 0.0

            num_vertical = types.count("VERTICAL")
            num_horizontal = types.count("HORIZONTAL")
            num_oblique = types.count("OBLIQUE")

            # ---------- SINGLE ----------
            if len(types) == 1:
                wall_score_edge += weights_sel[0] * 0.8

            # ---------- TWO ----------
            elif len(types) == 2:
                t1, t2 = types
                r1, r2 = weights_sel

                if t1 == "OBLIQUE" and t2 == "OBLIQUE":
                    corner_score_edge += (r1 + r2) * 1

                elif ("OBLIQUE" in types) and ("VERTICAL" in types):
                    wall_score_edge += (r1 + r2) * 0.6
                    corner_score_edge += (r1 + r2) * 0.3

                elif ("VERTICAL" in types) and ("HORIZONTAL" in types):
                    wall_score_edge += (r1 + r2) * 0.9

                else:
                    wall_score_edge += (r1 + r2) * 0.5

            # ---------- THREE ----------
            elif len(types) == 3:
                total_w = sum(weights_sel)

                if num_oblique >= 2 and num_vertical >= 1:
                    corner_score_edge += total_w * 1.0
                    wall_score_edge += total_w * 0.3

                elif num_oblique == 3:
                    corner_score_edge += total_w * 0.6
                    wall_score_edge += total_w * 0.2

                elif num_vertical >= 1 and num_horizontal >= 1:
                    wall_score_edge += total_w * 0.7
                    corner_score_edge += total_w * 0.4

                else:
                    wall_score_edge += total_w * 0.4

            # =========================
            # 🔥 confidence 반영
            # =========================

            edge_conf = sum(weights_sel)

            wall_score_edge *= edge_conf
            corner_score_edge *= edge_conf
        
            #cv2.imshow("depthmap",depth_map)

        #Fusion
        # EDGE
        edge_strength = wall_score_edge + corner_score_edge
        edge_reliability = edge_strength

        # DEPTH
        depth_strength = peak1_grad + peak2_grad
        depth_reliability = depth_strength

        # 정규화 (중요)
        total_rel = edge_reliability + depth_reliability + 1e-6
        edge_reliability /= total_rel
        depth_reliability /= total_rel

        # =========================
        # 🔥 1️⃣ dominant 판단
        # =========================

        margin = 0.25  # 민감도 (튜닝 가능)

        if edge_reliability > depth_reliability + margin:
            mode = "EDGE_DOMINANT"
        elif depth_reliability > edge_reliability + margin:
            mode = "DEPTH_DOMINANT"
        else:
            mode = "BALANCED"

        # =========================
        # 🔥 2️⃣ fusion 
        # =========================

        if mode == "EDGE_DOMINANT":
            # edge를 메인으로
            final_wall = wall_score_edge
            final_corner = corner_score_edge

            # depth는 보정만
            final_wall += 0.2 * wall_score_grad
            final_corner += 0.2 * corner_score_grad

        elif mode == "DEPTH_DOMINANT":
            # depth를 메인으로
            final_wall = wall_score_grad
            final_corner = corner_score_grad

            # edge는 보정만
            final_wall += 0.2 * wall_score_edge
            final_corner += 0.2 * corner_score_edge

        else:  # BALANCED
            # 둘을 균형 있게
            final_wall = 0.5 * wall_score_edge + 0.5 * wall_score_grad
            final_corner = 0.5 * corner_score_edge + 0.5 * corner_score_grad

        # =========================
        # 🔥 3️⃣ 최종 decision
        # =========================

        total = final_wall + final_corner + 1e-6
        corner_ratio = final_corner / total
        wall_ratio = final_wall / total

        if corner_ratio > 0.6:
            final_structure = "CORNER"

        elif wall_ratio > 0.6:
            final_structure = "WALL"

        else:
            final_structure = "UNDEFINED"

        if edge_vis is True: 
            display = cv2.resize(small_color, (256,256), interpolation=cv2.INTER_LINEAR)

            # 🔥 텍스트 설정
            text = f"{final_structure}"
    
            # 색상 (구조별로 다르게 하면 더 좋음)
            if final_structure == "WALL":
                color = (0, 255, 0)       # 초록
            elif final_structure == "CORNER":
                color = (0, 0, 255)       # 빨강
            else:
                color = (200, 200, 200)   # 회색

            # 🔥 텍스트 그리기
            cv2.putText(
                display,
                text,
                (10, 25),                  # 좌상단 위치
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,                       # 크기
                color,
                2,                         # 두께
                cv2.LINE_AA
            )

            cv2.imshow("lines", display)

        #----------------------------------------------------------------
        #hole_mask = object_mask.astype(np.uint8)  # 1 = hole

        #p5 = np.percentile(smoothed, 5)
        #p95 = np.percentile(smoothed, 95)

        #depth_geom = (smoothed - p5) / (p95 - p5 + 1e-6)
        #depth_geom = np.clip(depth_geom, 0, 1)

        #depth_filled = fill_depth_holes(
            #depth_geom,   
            #object_mask,
            #angle_grad,
            #depth_grad
        #)

        #h_d, w_d = depth_filled.shape
        #cx = w_d / 2

        #candidate_mask = structure_mask.copy()

        #low_grad = np.percentile(depth_grad, 40)
        #high_grad = np.percentile(depth_grad, 90)

        #grad_mask = (depth_grad > low_grad) & (depth_grad < high_grad)

        #candidate_mask &= grad_mask

        #edge_mask_small = cv2.resize(edges, (depth_norm.shape[1], depth_norm.shape[0]))
        #edge_mask_bool = edge_mask_small > 0

        #candidate_mask &= edge_mask_bool

        #y_center = int(h * 0.55)
        #band = int(h * 0.25)

        #center_mask = np.zeros_like(candidate_mask)
        #center_mask[y_center-band:y_center+band, :] = True

        #candidate_mask &= center_mask

        #mask_uint8 = (candidate_mask.astype(np.uint8) * 255)

        #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

        #regions = []

        #for i in range(1, num_labels):
            #area = stats[i, cv2.CC_STAT_AREA]

            #if area < 100:  # 너무 작은 영역 제거
                #continue

            #region_mask = (labels == i)

            # score 계산
            #mean_grad = np.mean(depth_grad[region_mask])
            #density = np.sum(region_mask) / (area + 1e-6)

            #score = mean_grad * density

            #regions.append((score, region_mask))

        #if len(regions) > 0:
            #regions = sorted(regions, key=lambda x: x[0], reverse=True)
            #best_region = regions[0][1]
        #else:
            #best_region = candidate_mask  # fallback

        #final_candidate_mask = best_region

        #ys, xs = np.where(final_candidate_mask)

        #points = []
        #grad_dirs = []

        #for y, x in zip(ys, xs):
            #z = depth_filled[y, x]

            # X
            #scale_x = 2.0 / w_d
            #X = (x - cx) * scale_x

            # Z
            #z_vals = depth_filled[ys, xs]
            #z_mean = np.mean(z_vals)

            #scale_z = 2.0   # 🔥 중요 (튜닝)

            #Z = (z - z_mean) * scale_z

            #points.append((X, Z))

            #dz = dzx[y, x] 

            #if abs(dz) > 1e-4:
                #dir_x = np.sign(x - cx)
                #dir_z = dz

                #inv_norm = 1.0 / np.sqrt(1.0 + dz*dz)
                #grad_dirs.append((dir_x * inv_norm, dir_z * inv_norm))

        # 이전 결과 저장용
        #if 'prev_points' not in locals() or prev_points is None:
            #prev_points = points
 
        
        # =========================
        # 🔥 WALL (수정됨)
        # =========================
        #if final_structure == "WALL":

            #if len(grad_dirs) < 10:
                #final_points = prev_points

            #else:
                # 1️⃣ 방향 RANSAC
                #direction = ransac_direction(grad_dirs)

                #if direction is None:
                    #final_points = prev_points
                #else:
                 #   # 2️⃣ 중심점
                  #  center_x = np.mean([p[0] for p in points])
                   # center_z = np.mean([p[1] for p in points])
                    #center = (center_x, center_z)
#
 #                   # 3️⃣ line 생성
  #                  line = direction_to_line(direction, center)
#
 #                   # 4️⃣ projection
  #                  final_points = project_points(points, line)
##                   prev_points = final_points
#
 #       # =========================
  #      # 🔥 CORNER (수정됨)
   ###
      #      if len(grad_dirs) < 20:
       #         final_points = prev_points
#
 #           else:
  ##
    ##               final_points = prev_points
#
 #               else:
  ####               dir1, dir2 = enforce_manhattan_dirs(dir1, dir2)

      #              # 중심
       #             center_x = np.mean([p[0] for p in points])
        #            center_z = np.mean([p[1] for p in points])
         #           center = (center_x, center_z)
#
 #                   # line 생성
  #                  line1 = direction_to_line(dir1, center)
   #                 line2 = direction_to_line(dir2, center)
#
 #                   # 🔥 각 점을 더 가까운 라인으로 할당
  #                  proj_points = []
#
 #                   for x, z in points:
#
 #                       d1 = abs(line1[0]*x + line1[1]*z + line1[2])
  #                      d2 = abs(line2[0]*x + line2[1]*z + line2[2])
#
 #                       if d1 < d2:
  #                          proj = project_points((x, z), line1)
   #                     else:
    #                        proj = project_points((x, z), line2)
#
 #                       proj_points.append(proj)
#
 #                   final_points = proj_points
  #                  prev_points = final_points
#
#
 #       # =========================
  #      # 🔥 UNDEFINED
   #     # =========================
    #    else:
     #       final_points = prev_points
#
 #       # =========================
  ##     # =========================
#
 #       canvas_h, canvas_w = 256, 256
  #      canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
#
 #       scale = 30.0   # 🔥 튜닝 (중요)
##       cx = canvas_w // 2
  ##
    #    if len(final_points) > 0:
     #       mean_x = np.mean([p[0] for p in final_points])
      #      mean_z = np.min([p[1] for p in final_points])
       # else:
        #    mean_x, mean_z = 0, 0
#
 #       for X, Z in final_points:
##           # 🔥 중심 정렬
  #          Xc = X - mean_x
   #         Zc = Z - mean_z
#
 #           px = int(cx + Xc * scale)
  #          pz = int(cz - Zc * scale)
#
 #           if 0 <= px < canvas_w and 0 <= pz < canvas_h:
  #              cv2.circle(canvas, (px, pz), 2, 255, -1)
#
 #       # 보기 좋게 blur (선택)
  ##
    #    cv2.imshow("point_cloud_xz", canvas_vis)

        # =========================
        # 🔥 4️⃣ 디버깅 출력 (강추)
        # =========================

        print("Mode:", mode)
        print("Edge rel:", edge_reliability)
        print("Depth rel:", depth_reliability)
        print("Final wall:", final_wall)
        print("Final corner:", final_corner)
        print("Final structure:", final_structure)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        


# ------------------- Main -------------------
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        print(f"Client connected: {addr}")

        # PC에서 보여줄 화면 크기
        target_H, target_W = 480, 640

        threading.Thread(target=receive_thread, args=(conn, raw_queue, gyro_queue, accel_queue), daemon=True).start()
        threading.Thread(target=depth_saliency_thread,daemon=True).start()

        # keep main alive
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__=="__main__":
    main()
