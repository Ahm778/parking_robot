"""Géométrie du parking : source unique pour le monde Gazebo, la carte Nav2, les places et l'environnement DRL.

Repère monde = repère map. x le long de l'allée, y vers la rangée A.

        y
        ^      mur (y = +9)
        |  [A1][A2][A3][A4][A5]      rangée A : places de y = +3 à +8,5
        |  ----butées (y = +3,25)----
  robot o------------------------->  allée centrale (y de -3 à +3)  -> x
        |  ----butées (y = -3,25)----
        |  [B1][B2][B3][B4][B5]      rangée B : places de y = -3 à -8,5
               mur (y = -9,3)
"""
XS = [-6.0, -3.0, 0.0, 3.0, 6.0]      # x du centre de chaque colonne de places (1 à 5)
ROW_SIGN = {'A': 1.0, 'B': -1.0}
SLOT_W, SLOT_L = 3.0, 5.5               # largeur, profondeur d'une place (m)
AISLE_HALF = 3.0                        # l'allée va de y = -3 à y = +3
SLOT_CY = AISLE_HALF + SLOT_L / 2       # |y| du centre d'une place = 5,75
CAR_CY = 6.0                            # |y| où l'on pose le centre d'une voiture
STOP_CY, STOP_SIZE = 3.25, (2.0, 0.15, 0.20)   # butées : 20 cm de haut, vues par le lidar (monté à ~13 cm)
WALL_C, WALL_T, WALL_H = 9.3, 0.2, 1.0  # murs : centre à ±9,3 m (le pick-up mesure 5,7 m), épaisseur, hauteur
ROBOT_START = (-8.0, 0.0)               # pose de départ du robot (= transformation map -> odom)


def slots():
    """Liste des places : id, rangée, numéro, centre (x, y)."""
    return [{'id': f'{r}{i + 1}', 'row': r, 'number': i + 1, 'x': x, 'y': s * SLOT_CY}
            for r, s in ROW_SIGN.items() for i, x in enumerate(XS)]


def walls():
    """Murs sous forme de boîtes (xmin, ymin, xmax, ymax)."""
    c, t = WALL_C, WALL_T / 2
    return [(-c - t, c - t, c + t, c + t), (-c - t, -c - t, c + t, -c + t),
            (c - t, -c, c + t, c), (-c - t, -c, -c + t, c)]


def stops():
    """Butées de parking (une par place) sous forme de boîtes (xmin, ymin, xmax, ymax)."""
    w, d = STOP_SIZE[0] / 2, STOP_SIZE[1] / 2
    return [(x - w, s * STOP_CY - d, x + w, s * STOP_CY + d) for s in ROW_SIGN.values() for x in XS]


# Distance du centre de chaque modèle à son extrémité côté allée (m), mesurée sur les maillages de gazebo_models
CAR_FRONT = {'hatchback': 2.34, 'hatchback_red': 2.34, 'hatchback_blue': 2.34,
             'suv': 2.58, 'suv_red': 2.58, 'pickup': 2.71}


def car_box(x, row):
    """Encombrement approximatif d'une voiture garée (pour l'environnement DRL)."""
    s = ROW_SIGN[row]
    y0, y1 = s * (CAR_CY - 2.5), s * (CAR_CY + 2.4)
    return (x - 1.1, min(y0, y1), x + 1.1, max(y0, y1))
