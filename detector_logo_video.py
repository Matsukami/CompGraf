# Detecção de logo em UM vídeo: tenta SIFT+FLANN; se não achar inliers suficientes,
# cai para Template Matching em bordas (Canny) com busca multi-escala e multi-rotação.
# Gera métricas de tempo de exposição, % de área e heatmap.
#
# Requisitos:
#   pip install opencv-contrib-python numpy
#
# Uso:
#   py detector_logo_video_nike.py

import cv2
import numpy as np
from pathlib import Path
import math

# ========================= PARÂMETROS =========================
VIDEO_PATH     = "meu_video.mp4"       # seu vídeo
TEMPLATE_PATH  = "logo_template.png"   # swoosh (fundo preto + swoosh branco funciona)
OUT_DIR        = Path("saida_detect")
OUT_DIR.mkdir(exist_ok=True)

# SIFT (primeira tentativa)
RATIO_TEST     = 0.80
INLIER_MIN     = 12
MIN_MATCH_FRAC = 0.02

# Desempenho
MAX_DIM        = 720     # redimensiona o frame para acelerar
FRAME_STRIDE   = 2       # processa 1 a cada N frames
DRAW_SAMPLE_N  = 2       # salva até N frames anotados

# Fallback por Template Matching em bordas
TM_EDGE_THRESH = 0.60    # correlação mínima p/ aceitar (0.55~0.75)
TM_SCALES      = np.linspace(0.4, 1.6, 13)   # faixas de escala (ajuste à sua cena)
TM_ANGLES      = list(range(-20, 21, 10))    # ângulos em graus (ajuste se houver rotação maior)
CANNY_T1       = 50
CANNY_T2       = 120

# ========================= FUNÇÕES AUXILIARES =========================
def preprocess_gray(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    blur = cv2.GaussianBlur(g, (0,0), 1.0)
    sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    return sharp

def edges(img_gray):
    e = cv2.Canny(img_gray, CANNY_T1, CANNY_T2)
    return e

def rotate_image(img, angle_deg):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    # calcula tamanho do novo canvas para não cortar
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - w/2
    M[1, 2] += (nH / 2) - h/2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=0)

def best_edge_template_match(frame_gray, templ_gray):
    """Busca em bordas multi-escala e multi-rotação. Retorna (score, (x,y,w,h), angle, scale)."""
    f_edge = edges(frame_gray)
    best = (0.0, None, 0, 1.0)

    for ang in TM_ANGLES:
        t_rot = rotate_image(templ_gray, ang)
        t_edge_rot = edges(t_rot)
        th, tw = t_edge_rot.shape[:2]
        if th < 8 or tw < 8:
            continue

        for sc in TM_SCALES:
            tw2 = max(8, int(tw * sc))
            th2 = max(8, int(th * sc))
            t_edge_rs = cv2.resize(t_edge_rot, (tw2, th2), interpolation=cv2.INTER_AREA)

            if t_edge_rs.shape[0] >= f_edge.shape[0] or t_edge_rs.shape[1] >= f_edge.shape[1]:
                continue

            res = cv2.matchTemplate(f_edge, t_edge_rs, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

            if maxVal > best[0]:
                best = (maxVal, (maxLoc[0], maxLoc[1], tw2, th2), ang, sc)

    return best  # score, bbox, angle, scale

# ========================= INICIALIZAÇÃO SIFT =========================
sift = cv2.SIFT_create(
    nfeatures=4000,
    contrastThreshold=0.02,
    edgeThreshold=5,
    sigma=1.2
)
index_params  = dict(algorithm=1, trees=5)
search_params = dict(checks=64)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ========================= TEMPLATE =========================
logo_bgr = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
if logo_bgr is None:
    raise FileNotFoundError(f"Não foi possível abrir TEMPLATE_PATH: {TEMPLATE_PATH}")

logo_gray_full = preprocess_gray(logo_bgr)
kp_logo, des_logo = sift.detectAndCompute(logo_gray_full, None)
if des_logo is None or len(kp_logo) < 4:
    # swoosh costuma cair aqui -> ainda assim vamos permitir fallback por TM
    des_logo = None
else:
    des_logo = des_logo.astype(np.float32)
    try:
        flann.add([des_logo]); flann.train()
    except Exception:
        pass

hL, wL = logo_gray_full.shape[:2]
logo_corners = np.float32([[0,0],[wL,0],[wL,hL],[0,hL]]).reshape(-1,1,2)
min_matches  = 8  # mínimo bruto para tentar homografia (se SIFT existir)

# Template para fallback: usamos grayscale puro (sem equalização extra) para bordas
logo_tm_gray = cv2.cvtColor(logo_bgr, cv2.COLOR_BGR2GRAY)

# ========================= VÍDEO =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Não foi possível abrir o vídeo: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

heatmap           = np.zeros((H, W), dtype=np.float32)
frames_total      = 0
frames_detectados = 0
area_pct_acum     = 0.0
amostras_salvas   = 0

# ========================= LOOP =========================
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    frames_total += 1

    if FRAME_STRIDE > 1 and (frames_total % FRAME_STRIDE) != 1:
        continue

    # ----- pré-processa / downscale para SIFT -----
    frame_gray_full = preprocess_gray(frame_bgr)
    Hcur, Wcur = frame_gray_full.shape[:2]
    scale = 1.0
    if max(Wcur, Hcur) > MAX_DIM:
        scale = float(MAX_DIM) / float(max(Wcur, Hcur))
    small_gray = cv2.resize(frame_gray_full, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_gray_full
    scale_up = 1.0 / scale

    detected = False
    poly = None

    # ======= TENTATIVA 1: SIFT (se houver descritores no template) =======
    if des_logo is not None:
        kp_f_small, des_f = sift.detectAndCompute(small_gray, None)
        if des_f is not None and len(kp_f_small) >= 4:
            matches_knn = flann.knnMatch(des_f.astype(np.float32), k=2)
            good = []
            for pair in matches_knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < RATIO_TEST * n.distance:
                    good.append(m)

            if len(good) >= max(min_matches, int(len(kp_logo) * MIN_MATCH_FRAC)):
                src_pts = np.float32([kp_logo[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([np.array(kp_f_small[m.queryIdx].pt, dtype=np.float32) * scale_up
                                      for m in good]).reshape(-1,1,2)
                Hmat, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
                if Hmat is not None and int(inlier_mask.sum()) >= INLIER_MIN:
                    proj = cv2.perspectiveTransform(logo_corners, Hmat).astype(np.int32)
                    poly = proj.reshape(-1,2)
                    detected = True

    # ======= TENTATIVA 2: TEMPLATE MATCHING EM BORDAS =======
    if not detected:
        # Para TM usamos o frame ORIGINAL em escala cheia (melhor precisão de bbox)
        frame_gray_tm = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        score, bbox, ang, sc = best_edge_template_match(frame_gray_tm, logo_tm_gray)

        if bbox is not None and score >= TM_EDGE_THRESH:
            x, y, w, h = bbox
            # caixa no espaço do frame -> polígono retangular
            poly = np.array([[x, y],
                             [x+w, y],
                             [x+w, y+h],
                             [x, y+h]], dtype=np.int32)
            detected = True

    # ======= ACUMULA MÉTRICAS / SALVA =======
    if detected and poly is not None:
        # máscara
        mask_logo = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_logo, [poly], 255)
        area_logo = float(cv2.countNonZero(mask_logo))
        area_pct  = (area_logo / float(H*W)) * 100.0

        frames_detectados += 1
        area_pct_acum     += area_pct
        heatmap += (mask_logo > 0).astype(np.float32)

        if amostras_salvas < DRAW_SAMPLE_N:
            annotated = frame_bgr.copy()
            cv2.polylines(annotated, [poly], True, (0,255,0), 2)
            cv2.imwrite(str(OUT_DIR / f"frame_detectado_{frames_total:06d}.jpg"), annotated)
            amostras_salvas += 1

    # log leve
    if frames_total % (100 * max(1, FRAME_STRIDE)) == 1:
        print(f"[debug] frame {frames_total} | detecções até agora: {frames_detectados}")

cap.release()

# ========================= MÉTRICAS =========================
tempo_exposicao_s = (frames_detectados * FRAME_STRIDE) / fps if fps > 0 else 0.0
area_pct_media    = (area_pct_acum / frames_detectados) if frames_detectados > 0 else 0.0

print("\n==== RESULTADOS ====")
print(f"Frames totais (lidos)    : {frames_total}")
print(f"Frames com detecção      : {frames_detectados}")
print(f"FPS do vídeo             : {fps:.2f}")
print(f"Tempo de exposição (s)   : {tempo_exposicao_s:.2f}")
print(f"% médio da área ocupada  : {area_pct_media:.4f}%")

# ========================= HEATMAP =========================
if heatmap.max() > 0:
    hm_norm  = (heatmap / heatmap.max() * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT_DIR / "heatmap.png"), hm_color)

    base = np.zeros((H, W, 3), dtype=np.uint8)
    exemplos = sorted(OUT_DIR.glob("frame_detectado_*.jpg"))
    if exemplos:
        base = cv2.imread(str(exemplos[0]))
        if base is None or base.shape[:2] != (H, W):
            base = np.zeros((H, W, 3), dtype=np.uint8)
    overlay = cv2.addWeighted(base, 0.6, hm_color, 0.4, 0.0)
    cv2.imwrite(str(OUT_DIR / "heatmap_sobreposto.png"), overlay)
else:
    cv2.imwrite(str(OUT_DIR / "heatmap.png"), np.zeros((H, W, 3), dtype=np.uint8))
    cv2.imwrite(str(OUT_DIR / "heatmap_sobreposto.png"), np.zeros((H, W, 3), dtype=np.uint8))
