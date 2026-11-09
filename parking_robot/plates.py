"""Plaques d'immatriculation tunisiennes : génération (monde Gazebo) et lecture OCR (vision).

Format réel (série normale) : fond noir, caractères blancs, de gauche à droite
    « série (3 chiffres) | تونس | numéro (4 chiffres) »   ex. 187 تونس 3985 ; plaque 52 cm x 11 cm.
Dans les commandes ROS, on écrit la plaque sans le mot arabe : « 187-3985 » (série-numéro).

En simulation, la plaque est agrandie à 0,70 m x 0,15 m (mêmes proportions) pour rester lisible par la
caméra 640 x 480 du TurtleBot3. Lecture : Tesseract en ligne de commande (paquet apt tesseract-ocr) sur les
chiffres seulement ; le mot « تونس », identique sur toutes les plaques, est ignoré.
"""
import difflib
import random
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

PLATE_W, PLATE_H, PLATE_Z = 0.70, 0.15, 0.55     # taille (m) et hauteur du centre au-dessus du sol
FONT_CANDIDATES = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                   '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
# « تونس » en formes de présentation arabes, déjà dans l'ordre d'affichage (gauche -> droite) :
# س final, ن initial, و final, ت initial. Rendu identique avec ou sans moteur de mise en forme (raqm).
TUNIS_GLYPHS = 'ﺲﻧﻮﺗ'
TUNIS = 'تونس'


def random_plate(rng):
    """Plaque au hasard, écrite « série-numéro », ex. '187-3985'."""
    return f'{rng.randint(100, 259)}-{rng.randint(1000, 9999)}'


def normalize(text):
    """Ne garde que les groupes de chiffres, dans l'ordre : '187 تونس 3985' ou '187-3985' -> '187|3985'."""
    return '|'.join(re.findall(r'\d+', text or ''))


def display(plate):
    """'187-3985' -> '187 تونس 3985' (pour l'affichage)."""
    groups = re.findall(r'\d+', plate)
    return f'{groups[0]} {TUNIS} {groups[1]}' if len(groups) == 2 else plate


def plate_similarity(read, target):
    """Similarité entre 0 et 1 (0,8 tolère environ un chiffre mal lu sur sept)."""
    a, b = normalize(read), normalize(target)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------- génération (monde)
def render_plate_png(plate, path, px_w=700, px_h=150):
    """Dessine la plaque : fond noir, liseré blanc, « série تونس numéro » en blanc."""
    from PIL import Image, ImageDraw, ImageFont
    serie, numero = re.findall(r'\d+', plate)[:2]
    img = Image.new('RGB', (px_w, px_h), 'black')
    d = ImageDraw.Draw(img)
    d.rectangle([7, 7, px_w - 8, px_h - 8], outline='white', width=3)
    font_path = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)

    def font(size):
        if font_path:
            return ImageFont.truetype(font_path, size, layout_engine=getattr(ImageFont, 'LAYOUT_BASIC', 0))
        return ImageFont.load_default(size=size)

    size = int(px_h * 0.72)
    while True:                                   # plus grande taille qui tient dans la plaque
        fd, fa = font(size), font(int(size * 0.80))
        parts = [(serie, fd), (TUNIS_GLYPHS, fa), (numero, fd)]
        boxes = [d.textbbox((0, 0), t, font=f) for t, f in parts]
        gap = size * 0.35
        total = sum(b[2] - b[0] for b in boxes) + 2 * gap
        height = max(b[3] - b[1] for b in boxes)
        if (total <= 0.90 * px_w and height <= 0.70 * px_h) or size <= 10:
            break
        size -= 2
    x = (px_w - total) / 2
    for (t, f), b in zip(parts, boxes):
        cy = px_h / 2 - (b[1] + b[3]) / 2           # chaque groupe centré verticalement
        d.text((x - b[0], cy), t, fill='white', font=f)
        x += (b[2] - b[0]) + gap
    img.save(path)


def write_plate_model(root, name, plate):
    """Crée root/name : un rectangle texturé (OBJ + MTL + PNG) qui regarde vers +x. Classic et Gazebo moderne."""
    m = Path(root) / name
    (m / 'meshes').mkdir(parents=True, exist_ok=True)
    (m / 'materials' / 'textures').mkdir(parents=True, exist_ok=True)
    render_plate_png(plate, m / 'meshes' / 'plate.png')
    render_plate_png(plate, m / 'materials' / 'textures' / 'plate.png')   # emplacement lu par Gazebo Classic
    w, h = PLATE_W / 2, PLATE_H / 2
    (m / 'meshes' / 'plate.mtl').write_text(
        'newmtl plate\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd plate.png\n')
    (m / 'meshes' / 'plate.obj').write_text(
        'mtllib plate.mtl\n'
        f'v 0 {-w} {-h}\nv 0 {w} {-h}\nv 0 {w} {h}\nv 0 {-w} {h}\n'
        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 1 0 0\nusemtl plate\nf 1/1/1 2/2/1 3/3/1 4/4/1\n')
    (m / 'model.config').write_text(
        f"<?xml version='1.0'?><model><name>{name}</name><version>1.0</version>"
        "<sdf version='1.6'>model.sdf</sdf><description>Plaque tunisienne générée</description></model>\n")
    (m / 'model.sdf').write_text(
        f"<?xml version='1.0'?><sdf version='1.6'><model name='{name}'><static>true</static>"
        f"<link name='link'><visual name='visual'><geometry><mesh><uri>model://{name}/meshes/plate.obj</uri>"
        "</mesh></geometry></visual></link></model></sdf>\n")


# ---------------------------------------------------------------- lecture (vision)
def find_plate(crop_bgr):
    """Cherche une plaque noire à caractères blancs dans la découpe d'une voiture. Renvoie l'image ou None.

    On cherche une zone sombre rectangulaire, allongée comme une plaque (52 x 11 cm, rapport ~4,7),
    qui contient une part raisonnable de pixels blancs (les caractères).
    """
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 20:
        return None
    big = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    value = big.max(axis=2)                       # luminosité HSV : une carrosserie rouge ou bleue n'est pas « noire »
    dark0 = (value < 80).astype(np.uint8) * 255
    # 1er essai avec fermeture (les caractères fins sont « bouchés » : la plaque forme un seul bloc sombre),
    # 2e essai sans (plaque proche ou cadre noir collé à la plaque)
    k = max(9, int(0.05 * big.shape[0]) | 1)
    closed = cv2.morphologyEx(dark0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    best, best_area = None, 0
    for dark in (closed, dark0):
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.minAreaRect(c)                      # rectangle orienté : tolère une plaque inclinée
            pts = cv2.boxPoints(rect)
            s_, d_ = pts.sum(1), np.diff(pts, axis=1).ravel()
            tl, br, tr, bl = pts[np.argmin(s_)], pts[np.argmax(s_)], pts[np.argmin(d_)], pts[np.argmax(d_)]
            rw, rh = float(np.linalg.norm(tr - tl)), float(np.linalg.norm(bl - tl))
            if rh < 12 or rw < 45 or not 2.8 < rw / rh < 7.5 or rw * rh <= best_area:
                continue
            M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]),
                                            np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]))
            roi_bgr = cv2.warpPerspective(big, M, (int(rw), int(rh)), flags=cv2.INTER_CUBIC)   # plaque redressée
            if roi_bgr.size == 0:
                continue
            roi = roi_bgr.max(axis=2)
            white, black = (roi > 120).mean(), (roi < 80).mean()
            if 0.04 < white < 0.5 and black > 0.45:
                best, best_area = roi_bgr, rw * rh
        if best is not None:
            break
    return best


def read_plate(plate_bgr, timeout_s=1.0):
    """Lit la série et le numéro (chiffres blancs sur fond noir). Renvoie 'série-numéro' ou ''."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    scale = 90.0 / max(gray.shape[0], 1)                                 # hauteur de travail ~90 px
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    text = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   # caractères en blanc
    H = text.shape[0]
    n, _, stats, _ = cv2.connectedComponentsWithStats(text, 8)
    # chiffres : composantes hautes et étroites ; on écarte le liseré, les points et le mot arabe (large)
    W = text.shape[1]
    cand = [s for s in stats[1:]
            if 0.15 * H < s[3] < 0.92 * H and 0.12 < s[2] / s[3] < 0.95
            and s[0] > 0 and s[1] > 0 and s[0] + s[2] < W and s[1] + s[3] < H]     # pas collé au bord
    if len(cand) < 3:
        return ''
    med = float(np.median([s[3] for s in cand]))
    digits = [s for s in cand if 0.75 * med < s[3] < 1.3 * med]                     # même hauteur que les chiffres
    if len(digits) < 3:
        return ''
    digits.sort(key=lambda s: s[0])
    gaps = [digits[i + 1][0] - (digits[i][0] + digits[i][2]) for i in range(len(digits) - 1)]
    cut = int(np.argmax(gaps)) + 1                                       # le mot arabe = le plus grand écart
    groups = [digits[:cut], digits[cut:]]
    # image propre : uniquement les chiffres, en noir sur blanc, les deux groupes séparés par un blanc
    parts = []
    for g in groups:
        x0, x1 = g[0][0], max(s[0] + s[2] for s in g)
        y0, y1 = min(s[1] for s in g), max(s[1] + s[3] for s in g)
        part = np.zeros_like(text[y0:y1, x0:x1])
        for s in g:
            sx, sy, sw, sh = s[0] - x0, s[1] - y0, s[2], s[3]
            part[sy:sy + sh, sx:sx + sw] = text[s[1]:s[1] + sh, s[0]:s[0] + sw]
        parts.append(part)
    hmax = max(p.shape[0] for p in parts)
    parts = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for p in parts]
    space = np.zeros((hmax, int(hmax * 0.8)), np.uint8)
    img = 255 - np.hstack([parts[0], space, parts[1]])
    img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    ok, png = cv2.imencode('.png', img)
    if not ok:
        return ''
    try:
        out = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '7',
                              '-c', 'tessedit_char_whitelist=0123456789 '],
                             input=png.tobytes(), capture_output=True, timeout=timeout_s, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ''
    groups_read = re.findall(r'\d+', out.stdout.decode(errors='ignore'))
    return '-'.join(groups_read[:2]) if len(groups_read) >= 2 else ''


if __name__ == '__main__':          # petit test : python3 -m parking_robot.plates
    p = random_plate(random.Random(0))
    render_plate_png(p, '/tmp/plate_test.png')
    print('plaque :', display(p), '-> lu :', read_plate(cv2.imread('/tmp/plate_test.png')))


# --- Lecture par les chiffres clairs : pour les voitures sombres, où la plaque se confond avec la carrosserie ---
def find_plate_chars(crop_bgr):
    """Plaque par ses chiffres clairs (pour les voitures sombres) : retourne la plaque redressée ou None."""
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 40:
        return None
    big = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    H, W = big.shape[:2]
    v = big.max(axis=2)
    k = max(31, min(int(0.5 * H) | 1, 101))
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    hi = float(np.percentile(th, 99.0))
    if hi < 20:
        return None
    binary = (th > max(15, 0.4 * hi)).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    comps = [stats[i] for i in range(1, n)
             if 0.06 * H < stats[i][3] < 0.6 * H and 0.12 < stats[i][2] / stats[i][3] < 1.0 and stats[i][4] > 60]
    if len(comps) < 5:
        return None
    # regroupement par chaînage : chiffres voisins de même hauteur, centres presque alignés (plaque un peu inclinée)
    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(comps):
        for j in range(i + 1, len(comps)):
            b = comps[j]
            ha, hb = a[3], b[3]
            hm = max(ha, hb)
            dx, dy = abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)), abs((a[1] + ha / 2) - (b[1] + hb / 2))
            if 0.75 < ha / hb < 1.33 and dx < 6.0 * hm and dy < 0.3 * hm + 0.3 * dx:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(comps)):
        groups.setdefault(find(i), []).append(comps[i])
    best = max(groups.values(), key=len)
    if len(best) < 5:
        return None
    cx = np.array([b[0] + b[2] / 2 for b in best], np.float32)
    cy = np.array([b[1] + b[3] / 2 for b in best], np.float32)
    ang = float(np.degrees(np.arctan(np.polyfit(cx, cy, 1)[0]))) if len(best) >= 3 else 0.0
    ang = max(-20.0, min(20.0, ang))
    M = cv2.getRotationMatrix2D((float(cx.mean()), float(cy.mean())), ang, 1.0)
    rot = cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    pts = []
    for b in best:
        for px, py in ((b[0], b[1]), (b[0] + b[2], b[1]), (b[0], b[1] + b[3]), (b[0] + b[2], b[1] + b[3])):
            pts.append(M @ np.array([px, py, 1.0]))
    pts = np.array(pts)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    hh = y1 - y0
    X0, X1 = int(max(0, x0 - 0.6 * hh)), int(min(W, x1 + 0.6 * hh))
    Y0, Y1 = int(max(0, y0 - 0.5 * hh)), int(min(H, y1 + 0.5 * hh))
    out = rot[Y0:Y1, X0:X1]
    return out if out.size else None


def read_plate_robust(crop_bgr):
    """Lit la plaque dans la zone basse d'une voiture : d'abord par les chiffres clairs (marche sur une
    carrosserie sombre), puis par le rectangle sombre d'origine. Retourne '' si rien n'est lu."""
    alt = find_plate_chars(crop_bgr)
    txt = read_plate(alt) if alt is not None else ''
    if txt:
        return txt
    found = find_plate(crop_bgr)
    return read_plate(found) if found is not None else ''
