"""Suivi de l'état de chaque place pendant la recherche (sans ROS : testable seul).

États d'une place :
  unseen    pas encore observée                                   (gris)
  seen      observée, sans indice ni pour ni contre               (gris clair)
  mismatch  ce n'est pas la voiture cherchée (plaque ou couleur)  (bleu)
  suspect   peut-être la bonne, mais plaque illisible d'ici       (orange)  -> le robot ira voir de près
  found     voiture trouvée                                       (vert)

Vie privée (cahier des charges § 4) : les plaques lues restent en mémoire pendant la mission, pour l'affichage,
et ne sont jamais écrites sur disque.
"""

COLORS = {'unseen': (0.55, 0.55, 0.55, 0.35), 'seen': (0.8, 0.8, 0.8, 0.45), 'mismatch': (0.15, 0.4, 1.0, 0.55),
          'suspect': (1.0, 0.55, 0.0, 0.7), 'found': (0.1, 0.85, 0.25, 0.85)}


class SlotTracker:
    def __init__(self, slots, plate_requested, color_requested, plate_threshold=0.8,
                 mismatch_votes=2, max_dist=4.0):
        self.plate_requested, self.color_requested = plate_requested, color_requested
        self.thr, self.mismatch_votes, self.max_dist = plate_threshold, mismatch_votes, max_dist
        self.slots = {s['id']: {**s, 'state': 'unseen', 'plate': '', 'match': 0, 'mismatch': 0, 'unread': 0}
                      for s in slots}

    def nearest(self, x, y):
        """Place la plus proche du point (x, y) de la carte, ou None si trop loin de toute place."""
        best = min(self.slots.values(), key=lambda s: (s['x'] - x) ** 2 + (s['y'] - y) ** 2)
        return best['id'] if ((best['x'] - x) ** 2 + (best['y'] - y) ** 2) ** 0.5 <= self.max_dist else None

    def observe(self, slot_id, obs):
        """obs : dict de la vision (color_score, plate_read, plate_score). Met à jour l'état de la place."""
        s = self.slots[slot_id]
        if s['state'] == 'found':
            return
        sc, read, ps = obs.get('color_score'), obs.get('plate_read') or '', obs.get('plate_score')
        color_ok = (not self.color_requested) or (sc is not None and sc >= 0.4)
        color_bad = self.color_requested and sc is not None and sc < 0.2
        if read:
            s['plate'] = read
        if self.plate_requested:
            if not obs.get('ocr_tried', True):
                return                            # plaque pas encore tentée : aucune information
            if read and ps is not None and ps >= self.thr:
                s['match'] += 1
            elif (read and ps is not None and ps < 0.5) or color_bad:
                s['mismatch'] += 1
            elif color_ok:
                s['unread'] += 1                 # plaque illisible ou douteuse, couleur compatible
        else:
            if sc is not None and sc >= 0.6:
                s['match'] += 1
            elif color_bad:
                s['mismatch'] += 1
            else:
                s['unread'] += 1
        self._update(s)

    def _update(self, s):
        if s['mismatch'] >= self.mismatch_votes and s['match'] == 0:
            s['state'] = 'mismatch'
        elif s['unread'] > 0 or s['match'] > 0:
            s['state'] = 'suspect'
        else:
            s['state'] = 'seen'

    def set_found(self, slot_id):
        self.slots[slot_id]['state'] = 'found'

    def mark_inspected_unreadable(self, slot_id):
        """Vu de près sans conclusion : on ne le reproposera plus."""
        s = self.slots[slot_id]
        if s['state'] == 'suspect':
            s['state'] = 'seen'
            s['unread'], s['match'] = 0, 0

    def suspects(self):
        return [sid for sid, s in self.slots.items() if s['state'] == 'suspect']

    def summary(self):
        keys = ('id', 'row', 'number', 'x', 'y', 'state', 'plate')
        return {sid: {k: s[k] for k in keys} for sid, s in self.slots.items()}
