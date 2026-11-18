#drawing of the shift register encoder and of the Meggitt decoder (4.2.2, questions 4 and 6)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def bascule(ax, x, y, nom):
    ax.add_patch(Rectangle((x, y - 0.4), 1.2, 0.8, fc="#e7f1fb", ec="#333333", lw=1.5))
    ax.text(x + 0.6, y, nom, ha="center", va="center", fontsize=11)

def xor(ax, x, y):
    ax.add_patch(Circle((x, y), 0.2, fc="white", ec="#333333", lw=1.5, zorder=3))
    ax.text(x, y, "+", ha="center", va="center", fontsize=13, zorder=4)

def fil(ax, xs, ys, fleche=True):
    ax.plot(xs[:-1] if fleche else xs, ys[:-1] if fleche else ys, color="#333333", lw=1.4)
    if fleche:
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))

def noeud(ax, x, y):
    ax.plot([x], [y], "o", color="#333333", ms=5)

def interrupteur(ax, x, y, texte):
    ax.plot([x - 0.3, x + 0.25], [y, y + 0.22], color="#c0392b", lw=2)
    ax.plot([x - 0.3, x + 0.3], [y, y], "o", color="#c0392b", ms=4)
    ax.text(x, y + 0.42, texte, ha="center", color="#c0392b", fontsize=10)

#encoder
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.text(-0.1, 2.2, "m(x)\nm3 d'abord", ha="center", va="bottom", fontsize=10)
fil(ax, [0, 1.3], [2, 2])
xor(ax, 1.5, 2)
fil(ax, [1.7, 2.2], [2, 2], False)
interrupteur(ax, 2.5, 2, "I1")
fil(ax, [2.8, 4], [2, 2])
noeud(ax, 3.3, 2)
bascule(ax, 4, 2, "D  s0")
fil(ax, [5.2, 5.8], [2, 2])
xor(ax, 6, 2)
fil(ax, [6.2, 6.7], [2, 2])
bascule(ax, 6.7, 2, "D  s1")
fil(ax, [7.9, 8.5], [2, 2])
bascule(ax, 8.5, 2, "D  s2")
fil(ax, [9.7, 10.3, 10.3, 1.5, 1.5], [2, 2, 3.4, 3.4, 2.2])
noeud(ax, 10.3, 2)
fil(ax, [3.3, 3.3, 6, 6], [2, 1.1, 1.1, 1.8])
ax.text(4.6, 0.8, "f = m + s2", ha="center", fontsize=10)
#output
noeud(ax, 0.7, 2)
fil(ax, [0.7, 0.7, 11.2], [2, 0.2, 0.2])
fil(ax, [10.3, 11.2], [2, 2])
ax.plot([11.8, 11.25], [1.1, 2.0], color="#c0392b", lw=2)
ax.plot([11.2, 11.2, 11.8], [2, 0.2, 1.1], "o", color="#c0392b", ms=4)
ax.text(12.1, 1.75, "I2", color="#c0392b", fontsize=10)
fil(ax, [11.8, 13], [1.1, 1.1])
ax.text(13.1, 1.1, "c", va="center", fontsize=12)
ax.text(11.0, 0.35, "message", ha="right", fontsize=9)
ax.text(11.0, 2.15, "redondance", ha="right", fontsize=9)
ax.text(6.2, -0.45, "Coups 1 à 4 : I1 fermé, I2 sur message.   Coups 5 à 7 : I1 ouvert (f = 0), I2 sur redondance.", ha="center", fontsize=10)
ax.set_title("Codeur systématique du C(7,4), g(x) = 1 + x + x³")
ax.set_xlim(-0.8, 13.6); ax.set_ylim(-0.7, 3.8); ax.axis("off")
plt.savefig("figures/fig12_codeur_registre.png", dpi=150, bbox_inches="tight")

#Meggitt decoder
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.text(0, 3.2, "r(x)\nr6 d'abord", ha="center", va="bottom", fontsize=10)
fil(ax, [0, 1.3], [3, 3])
noeud(ax, 0.7, 3)
xor(ax, 1.5, 3)
fil(ax, [1.7, 2.4], [3, 3])
bascule(ax, 2.4, 3, "D  s0")
fil(ax, [3.6, 4.3], [3, 3])
xor(ax, 4.5, 3)
fil(ax, [4.7, 5.3], [3, 3])
bascule(ax, 5.3, 3, "D  s1")
fil(ax, [6.5, 7.1], [3, 3])
bascule(ax, 7.1, 3, "D  s2")
fil(ax, [8.3, 8.9, 8.9, 1.5, 1.5], [3, 3, 4.3, 4.3, 3.2])
fil(ax, [4.5, 4.5], [4.3, 3.2]); noeud(ax, 4.5, 4.3)
noeud(ax, 8.9, 3)
#detector
ax.add_patch(Rectangle((3.6, 1.45), 3.4, 0.7, fc="#f9eeda", ec="#333333", lw=1.5))
ax.text(5.3, 1.8, "détecteur : (s0, s1, s2) = (1, 0, 1)", ha="center", va="center", fontsize=10)
fil(ax, [3.0, 3.0, 3.6], [2.6, 1.8, 1.8])
fil(ax, [5.9, 5.9], [2.6, 2.15])
fil(ax, [7.7, 7.7, 7.0], [2.6, 1.8, 1.8])
#buffer
fil(ax, [0.7, 0.7, 1.8], [3, 0.5, 0.5])
ax.add_patch(Rectangle((1.8, 0.1), 6.6, 0.8, fc="#e7f1fb", ec="#333333", lw=1.5))
ax.text(5.1, 0.5, "tampon à décalage de 7 bits", ha="center", va="center", fontsize=10)
fil(ax, [8.4, 9.8], [0.5, 0.5])
xor(ax, 10, 0.5)
fil(ax, [10.2, 11.2], [0.5, 0.5])
ax.text(11.3, 0.5, "mot corrigé", va="center", fontsize=11)
fil(ax, [5.3, 5.3, 10, 10], [1.45, 1.1, 1.1, 0.7])
ax.text(8.5, 1.2, "correction", fontsize=9)
ax.set_title("Décodeur de Meggitt du C(7,4) : 7 coups de calcul, puis 7 coups de sortie, entrée à 0")
ax.set_xlim(-0.8, 12.8); ax.set_ylim(-0.1, 4.7); ax.axis("off")
plt.savefig("figures/fig13_meggitt.png", dpi=150, bbox_inches="tight")
