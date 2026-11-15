"""Géométrie du parking : source unique pour le monde Gazebo, la carte Nav2, les places et l'environnement DRL.

Repère monde = repère map. x le long des allées, y vers la rangée A.

Mode normal (campagne finale) : 2 rangées face à face, une allée.

        y
        ^      mur (y = +9,3)
        |  [A1][A2][A3][A4][A5]      rangée A : places de y = +3 à +8,5
        |  ----butées (y = +3,25)----
  robot o------------------------->  allée centrale (y de -3 à +3)  -> x
        |  ----butées (y = -3,25)----
        |  [B1][B2][B3][B4][B5]      rangée B : places de y = -3 à -8,5
               mur (y = -9,3)

Mode démo (export PARKING_DEMO=1, pour les vidéos) : 4 rangées, 2 allées et une voie qui fait le tour.

   ┌──────────── voie périphérique ────────────┐
   │   [A1][A2][A3][A4][A5]                     │
   │ ═════════════ allée 1 (y = +9,7) ═════════ │
   │   [B1][B2][B3][B4][B5]                     │
   │   ─────────── terre-plein ───────────      │
   │   [C1][C2][C3][C4][C5]                     │
   │ ═════════════ allée 2 (y = -9,7) ═════════ │
   │   [D1][D2][D3][D4][D5]                     │
   └────────────────────────────────────────────┘
   Le robot part de la voie de gauche, parcourt l'allée 1 puis l'allée 2 (parcours en S).
"""
import os

DEMO = os.environ.get('PARKING_DEMO') == '1'

SLOT_L = 5.5                                    # profondeur d'une place (m)
STOP_SIZE = (2.0, 0.15, 0.20)                   # butées : 20 cm de haut, vues par le lidar (monté à ~13 cm)
WALL_T, WALL_H = 0.2, 1.0                       # murs : épaisseur, hauteur
CAR_TO_AISLE_NORMAL = 6.0                       # distance centre d'allée -> centre de voiture (mode normal)
SPEED = 3.0 if DEMO else 1.0                    # démo : jusqu'à 3 x 0,22 = 0,66 m/s en ligne droite dégagée

if not DEMO:
    XS = [-6.0, -3.0, 0.0, 3.0, 6.0]            # x du centre de chaque colonne de places (1 à 5)
    SLOT_W = 3.0                                # largeur d'une place (m)
    AISLE_HALF = 3.0                            # l'allée va de y = -3 à y = +3
    AISLES_Y = [0.0]                            # y du centre de chaque allée, dans l'ordre de parcours
    # rangée -> (allée, côté : +1 si la rangée est au-dessus de son allée)
    ROW_LAYOUT = {'A': (0.0, 1.0), 'B': (0.0, -1.0)}
    WALL_X = WALL_Y = 9.3                       # murs à ±9,3 m (le pick-up mesure 5,7 m)
    ROBOT_START = (-8.0, 0.0)                   # pose de départ du robot (= transformation map -> odom)
    PED = (-1.5, -2.5, 2.5)                     # piéton : x de traversée, y_min, y_max
    BLOCKER = (1.5, 0.0)                        # obstacle soudain du scénario S4
    BLOCKER_PARK = (0.0, -15.0)                 # où la caisse attend, hors du parking
    MEDIANS = []                                # terre-pleins (boîtes x, y, sx, sy)
else:
    XS = [-7.0, -3.5, 0.0, 3.5, 7.0]
    SLOT_W = 3.5
    AISLE_HALF = 3.5                            # allées de 7 m
    MEDIAN_HALF = 0.7                           # terre-plein central de 1,4 m entre les rangées B et C
    _ay = MEDIAN_HALF + SLOT_L + AISLE_HALF     # 9,7 m
    AISLES_Y = [_ay, -_ay]
    ROW_LAYOUT = {'A': (_ay, 1.0), 'B': (_ay, -1.0), 'C': (-_ay, 1.0), 'D': (-_ay, -1.0)}
    WALL_X = XS[-1] + SLOT_W / 2 + 4.75         # voie périphérique de 4,5 m à gauche et à droite : 13,5 m
    WALL_Y = _ay + AISLE_HALF + SLOT_L + 4.3    # voie périphérique de ~4 m en haut et en bas : 23,0 m
    ROBOT_START = (-11.25, 0.0)                 # dans la voie de gauche, à mi-hauteur
    PED = (-1.75, _ay - 2.5, _ay + 2.5)         # le piéton traverse l'allée 1
    BLOCKER = (1.75, _ay)                       # S4 : caisse dans l'allée 1
    BLOCKER_PARK = (60.0, 60.0)                 # hors du sol et du champ de la caméra
    MEDIANS = [(0.0, 0.0, XS[-1] - XS[0] + SLOT_W, 0.3)]          # bordure de 30 cm de large


def _row(aisle_y, side):
    slot_cy = aisle_y + side * (AISLE_HALF + SLOT_L / 2)
    return {'aisle_y': aisle_y, 'side': side, 'slot_cy': slot_cy,
            'car_cy': slot_cy + side * 0.25,                         # voiture enfoncée de 25 cm dans sa place
            'stop_y': aisle_y + side * (AISLE_HALF + 0.25)}


ROWS = {r: _row(*v) for r, v in ROW_LAYOUT.items()}
ROW_SIGN = {r: v[1] for r, v in ROW_LAYOUT.items()}               # +1 : rangée au-dessus de son allée
ROW_AISLE = {r: (v[0], v[1]) for r, v in ROW_LAYOUT.items()}       # rangée -> (y de l'allée, côté)
LOOK_OFF = (AISLE_HALF + SLOT_L / 2 + 0.25) - CAR_TO_AISLE_NORMAL # décalage des points d'arrêt (0 en normal)
SLOT_CY = ROWS['A']['slot_cy'] - ROWS['A']['aisle_y']             # (compatibilité) |y| relatif à l'allée
CAR_CY = ROWS['A']['car_cy'] - ROWS['A']['aisle_y']
STOP_CY = ROWS['A']['stop_y'] - ROWS['A']['aisle_y']
WALL_C = max(WALL_X, WALL_Y)


def slots():
    """Liste des places : id, rangée, numéro, centre (x, y)."""
    return [{'id': f'{r}{i + 1}', 'row': r, 'number': i + 1, 'x': x, 'y': ROWS[r]['slot_cy']}
            for r in ROWS for i, x in enumerate(XS)]


def walls():
    """Murs sous forme de boîtes (xmin, ymin, xmax, ymax)."""
    cx, cy, t = WALL_X, WALL_Y, WALL_T / 2
    return [(-cx - t, cy - t, cx + t, cy + t), (-cx - t, -cy - t, cx + t, -cy + t),
            (cx - t, -cy, cx + t, cy), (-cx - t, -cy, -cx + t, cy)]


def stops():
    """Butées de parking (une par place) sous forme de boîtes (xmin, ymin, xmax, ymax)."""
    w, d = STOP_SIZE[0] / 2, STOP_SIZE[1] / 2
    return [(x - w, ROWS[r]['stop_y'] - d, x + w, ROWS[r]['stop_y'] + d) for r in ROWS for x in XS]


def map_obstacles():
    """Ce que la carte Nav2 connaît d'avance : murs, butées, et en mode démo terre-plein + rangées
    (le robot n'entre jamais dans les places : le planificateur passe alors par les allées et la voie périphérique)."""
    obs = walls() + stops()
    if DEMO:
        x0, x1 = XS[0] - SLOT_W / 2, XS[-1] + SLOT_W / 2
        # tout l'îlot central (rangées B et C + terre-plein) est interdit : sinon le planificateur voit un
        # couloir de 50 cm entre la bordure et les places et y envoie le robot, qui reste coincé
        obs.append((x0, -MEDIAN_HALF - 0.1, x1, MEDIAN_HALF + 0.1))
        for r in ROWS.values():
            y0 = r['stop_y'] + r['side'] * 0.1
            y1 = r['aisle_y'] + r['side'] * (AISLE_HALF + SLOT_L)
            obs.append((x0, min(y0, y1), x1, max(y0, y1)))
    return obs


# Distance du centre de chaque modèle à son extrémité côté allée (m), mesurée sur les maillages de gazebo_models
CAR_FRONT = {'hatchback': 2.34, 'hatchback_red': 2.34, 'hatchback_blue': 2.34,
             'suv': 2.58, 'suv_red': 2.58, 'pickup': 2.71}


def car_box(x, row):
    """Encombrement approximatif d'une voiture garée (pour l'environnement DRL)."""
    r = ROWS[row]
    y0, y1 = r['car_cy'] - r['side'] * 2.5, r['car_cy'] + r['side'] * 2.4
    return (x - 1.1, min(y0, y1), x + 1.1, max(y0, y1))
