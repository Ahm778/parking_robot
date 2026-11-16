"""Lecture FIABLE des plaques de la maquette (robot réel), sans Tesseract.

Principe : les plaques de la démo sont imprimées par nous, avec une police connue (DejaVu Sans Bold, la même
que dans la simulation). Au lieu d'un OCR généraliste, on compare chaque chiffre à des modèles de cette
police : c'est beaucoup plus sûr et plus rapide. Et on préfère « je ne sais pas » à une erreur :

1. trouver_plaques : rectangles sombres à 4 coins, allongés comme une plaque (rapport 3 à 7) ;
2. redresser       : perspective corrigée -> image 470 x 100 (plaque vue de face, même inclinée à 45°) ;
3. chiffres        : composantes blanches de même hauteur, séparées en deux groupes par le mot arabe ;
4. reconnaître     : chaque chiffre est comparé aux modèles ; s'il n'est pas clairement reconnu
                     (score trop bas, ou deux chiffres trop proches), toute la lecture est rejetée ;
   + format imposé (3 chiffres - 4 chiffres, celui de nos plaques) : une lecture incomplète est refusée ;
5. LecteurFiable   : vote sur plusieurs images ; une plaque n'est « confirmée » que si la même lecture
                     revient au moins 3 fois sur les 7 dernières images, et sans lecture concurrente.

Les modèles de chiffres sont dans scripts/data/chiffres_plaque.npz (fabriqués par
scripts/imprimer_plaques.py). Compatible Python 3.6, numpy 1.13, OpenCV 4.1 (Jetson, JetPack 4.6).
"""
import os
from collections import Counter, deque

import cv2
import numpy as np

W_STD, H_STD = 470, 100            # plaque redressée (rapport 4,7 comme la vraie : 52 x 11 cm)
TW, TH = 24, 36                     # taille d'un chiffre normalisé
MODELES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'scripts', 'data', 'chiffres_plaque.npz')


def normaliser_chiffre(binaire):
    """Chiffre blanc sur noir (découpé à sa boîte) -> image TW x TH en float, centrée, proportions gardées."""
    h, w = binaire.shape
    s = (TH - 4) / float(h)
    nw = max(1, min(TW - 2, int(round(w * s))))
    petit = cv2.resize(binaire, (nw, TH - 4), interpolation=cv2.INTER_AREA)
    out = np.zeros((TH, TW), np.float32)
    x0 = (TW - nw) // 2
    out[2:TH - 2, x0:x0 + nw] = petit
    return out


class Modeles:
    def __init__(self, chemin=MODELES):
        d = np.load(chemin)
        x = d['images'].astype(np.float32).reshape(len(d['labels']), -1)
        x -= x.mean(1, keepdims=True)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
        self.x, self.labels = x, d['labels'].astype(int)

    def reconnaitre(self, chiffre):
        """Renvoie (chiffre, score, marge) ; score = corrélation normalisée (1 = identique)."""
        v = chiffre.reshape(-1).astype(np.float32)
        v = v - v.mean()
        v /= (np.linalg.norm(v) + 1e-6)
        sc = self.x.dot(v)
        meilleur = {}
        for lab, s in zip(self.labels, sc):
            if s > meilleur.get(lab, -2):
                meilleur[lab] = s
        ordre = sorted(meilleur.items(), key=lambda t: -t[1])
        return ordre[0][0], float(ordre[0][1]), float(ordre[0][1] - ordre[1][1])


def _ordonner(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def trouver_plaques(img, largeur_min_frac=0.06):
    """Quadrilatères candidats (coins : haut-gauche, haut-droit, bas-droit, bas-gauche), du plus grand au plus petit."""
    h, w = img.shape[:2]
    v = img.max(axis=2) if img.ndim == 3 else img
    v = cv2.GaussianBlur(v, (5, 5), 0)
    masques = [cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                     max(31, (min(h, w) // 6) | 1), 10),
               cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 80) | 1, max(3, w // 80) | 1))
    quads = []
    for m in masques:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)          # bouche les chiffres : la plaque = un seul bloc
        contours = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in contours:
            if cv2.contourArea(c) < (largeur_min_frac * w) ** 2 / 6:
                continue
            per = cv2.arcLength(c, True)
            q = cv2.approxPolyDP(c, 0.03 * per, True)
            if len(q) != 4 or not cv2.isContourConvex(q):
                q = cv2.boxPoints(cv2.minAreaRect(c)).reshape(4, 1, 2)
            q = _ordonner(q)
            larg = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            haut = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            if haut < 8 or larg < largeur_min_frac * w or not 2.6 < larg / haut < 8.0:
                continue
            if any(np.abs(q - p).max() < 6 for p in quads):
                continue
            quads.append(q)
    quads.sort(key=lambda q: -cv2.contourArea(q.reshape(4, 1, 2)))
    return quads[:6]


def redresser(img, q):
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W_STD, 0], [W_STD, H_STD], [0, H_STD]]))
    return cv2.warpPerspective(img, M, (W_STD, H_STD), flags=cv2.INTER_CUBIC)


def chiffres(plaque):
    """Plaque redressée -> (groupe gauche, groupe droit) de chiffres binaires, ou None si ce n'est pas une plaque."""
    v = plaque.max(axis=2) if plaque.ndim == 3 else plaque
    b = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if not 0.05 < (b > 0).mean() < 0.55:                     # surtout noir, avec des caractères blancs
        return None
    n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
    comp = [s for s in st[1:]
            if 0.35 * H_STD < s[3] < 0.92 * H_STD and 0.12 < s[2] / float(s[3]) < 1.0
            and s[0] > 2 and s[0] + s[2] < W_STD - 2 and s[1] > 1 and s[1] + s[3] < H_STD - 1]
    if len(comp) < 3:
        return None
    hmed = float(np.median([s[3] for s in comp]))
    dig = sorted([s for s in comp if abs(s[3] - hmed) < 0.2 * hmed], key=lambda s: s[0])
    if not 3 <= len(dig) <= 7:
        return None
    bas = np.median([s[1] + s[3] for s in dig])
    if any(abs(s[1] + s[3] - bas) > 0.2 * hmed for s in dig):   # chiffres alignés sur une même ligne
        return None
    ecarts = [dig[i + 1][0] - (dig[i][0] + dig[i][2]) for i in range(len(dig) - 1)]
    i = int(np.argmax(ecarts))
    if ecarts[i] < 1.2 * hmed:                                    # pas de place pour le mot « تونس »
        return None
    g, d = dig[:i + 1], dig[i + 1:]
    if not (1 <= len(g) <= 3 and 1 <= len(d) <= 4):
        return None
    decoupe = lambda s: normaliser_chiffre(b[s[1]:s[1] + s[3], s[0]:s[0] + s[2]])
    return [decoupe(s) for s in g], [decoupe(s) for s in d]


class LecteurFiable:
    """Lecture image par image + vote. Paramètres choisis pour ne jamais confirmer une mauvaise plaque."""

    def __init__(self, score_min=0.72, marge_min=0.06, fenetre=7, votes=3, format_plaque=(3, 4)):
        self.modeles = Modeles()
        self.format = format_plaque          # nombre de chiffres (série, numéro) ; None = libre
        self.score_min, self.marge_min, self.votes = score_min, marge_min, votes
        self.hist = deque(maxlen=fenetre)

    def lire_image(self, img):
        """Une seule image -> (texte 'série-numéro' ou '', quadrilatère ou None). '' = pas sûr."""
        for q in trouver_plaques(img):
            groupes = chiffres(redresser(img, q))
            if groupes is None:
                continue
            if self.format and (len(groupes[0]), len(groupes[1])) != tuple(self.format):
                return '', q                                      # un chiffre manque ou est en trop : refusé
            txt = []
            for grp in groupes:
                s = ''
                for c in grp:
                    lab, score, marge = self.modeles.reconnaitre(c)
                    if score < self.score_min or marge < self.marge_min:
                        return '', q                              # un chiffre douteux : on refuse tout
                    s += str(lab)
                txt.append(s)
            return '-'.join(txt), q
        return '', None

    def mettre_a_jour(self, img):
        """Ajoute une image au vote. Renvoie (plaque confirmée ou '', lecture de cette image, quadrilatère)."""
        lu, q = self.lire_image(img)
        self.hist.append(lu)
        c = Counter(x for x in self.hist if x)
        confirme = ''
        if c:
            (top, n), = c.most_common(1)
            autres = sum(c.values()) - n
            if n >= self.votes and autres <= n // 3:
                confirme = top
        return confirme, lu, q

    def reinitialiser(self):
        self.hist.clear()
