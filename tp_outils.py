"""
tp_outils.py — fonctions FOURNIES pour le TP « Chaîne de transmission
numérique » (ENSIM 5A ASTRE).  Ne pas modifier.

Contient :
  * quelques utilitaires (Q, H2, teb, intervalle de confiance, tracés) ;
  * les paramètres personnalisés de chaque étudiant ;
  * les fonctions `verifie_*` qui testent VOS fonctions.  Un test qui passe
    affiche « OK » ; sinon il affiche ce qui ne va pas (AssertionError).
"""
import hashlib
import unicodedata
from math import erfc, sqrt

import numpy as np
import matplotlib.pyplot as plt


# ===================================================================== utilitaires
def Q(x):
    """Q(x) = P(N(0,1) > x), vectorisée."""
    x = np.asarray(x, dtype=float)
    return 0.5 * np.vectorize(erfc)(x / sqrt(2))


def H2(p):
    """Entropie binaire (bit/symbole)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.nan_to_num(h)

# Compute the Shannon entropy (in bits) of a probability distribution
def entropy(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy=-np.sum(p * np.log2(p))
    return np.nan_to_num(entropy)   


def teb(b_emis, b_recus):
    """Rend (TEB, nombre d'erreurs) entre deux suites de bits."""
    b_emis, b_recus = np.asarray(b_emis), np.asarray(b_recus)
    n = min(len(b_emis), len(b_recus))
    err = int(np.count_nonzero(b_emis[:n] != b_recus[:n]))
    return err / n, err


def intervalle_confiance(teb_mesure, n_bits, z=1.96):
    """Intervalle de confiance à 95 % (approximation gaussienne)."""
    s = np.sqrt(np.maximum(teb_mesure * (1 - teb_mesure), 1e-300) / n_bits)
    return np.maximum(teb_mesure - z * s, 0), teb_mesure + z * s


def figure_teb(titre="TEB en fonction de Eb/N0"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("TEB")
    ax.set_title(titre)
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def trace_constellation(ax, z, titre="", nmax=3000):
    z = np.asarray(z)[:nmax]
    ax.plot(z.real, z.imag, ".", ms=2, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(titre)


def trace_oeil(ax, y, L, debut, nb_traces=300, titre="Diagramme de l'œil"):
    """y : signal après filtre de réception ; L : échantillons/symbole ;
    debut : indice du premier instant d'échantillonnage optimal."""
    t = np.arange(-L, L + 1) / L
    for k in range(nb_traces):
        i = debut + k * L
        if i - L < 0 or i + L + 1 > len(y):
            break
        ax.plot(t, np.real(y[i - L : i + L + 1]), "b", lw=0.4, alpha=0.3)
    ax.set_xlabel("t / T")
    ax.set_title(titre)
    ax.grid(True, alpha=0.3)


# ===================================================================== paramètres
def parametres_perso(nom):
    """Paramètres personnels, déterminés par votre NOM (tel qu'il figure sur
    la liste d'appel).  Vos résultats numériques doivent correspondre à ces
    valeurs : deux étudiants n'ont pas les mêmes."""
    n = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode().upper().strip()
    h = int(hashlib.sha256(n.encode()).hexdigest(), 16)
    r = np.random.default_rng(h % (2**32))
    P = {
        "nom": n,
        "p1": float(np.round(r.uniform(0.15, 0.40), 2)),        # P(bit = 1)
        "pe_cbs": float(np.round(r.uniform(0.02, 0.15), 3)),     # CBS
        "A": 1.0,                                               # amplitude
        "a_unif": float(np.round(r.uniform(1.2, 2.0), 2)),       # bruit U[-a,a]
        "sigma": float(np.round(r.uniform(0.5, 0.9), 2)),        # bruit gaussien
        "L": int(r.choice([4, 8, 16])),                          # éch./bit
        "alpha": float(r.choice([0.22, 0.25, 0.35])),            # roll-off
        "beta": float(np.round(r.uniform(0.2, 0.5), 2)),         # 2e trajet
        "retard_trajet": int(r.choice([1, 2])),                  # en symboles (T)
        "graine": int(h % 10**6),
        "p1_ext": float(np.round(r.uniform(0.05, 0.15), 2)),   # source très déséquilibrée (2.3, 2.5)
    }
    return P


# ===================================================================== tests
def _ok(msg):
    print(f"OK  — {msg}")


def verifie_source(source):
    rng = np.random.default_rng(0)
    for p1 in (0.5, 0.2, 0.9):
        b = np.asarray(source(200_000, p1, rng))
        assert b.shape == (200_000,), "la source doit rendre un vecteur de N bits"
        assert set(np.unique(b)) <= {0, 1}, "la source doit rendre des 0 et des 1"
        f = b.mean()
        tol = 5 * np.sqrt(p1 * (1 - p1) / len(b))
        assert abs(f - p1) < tol, f"fréquence des 1 = {f:.4f}, attendu {p1} ± {tol:.4f}"
    _ok("source binaire")


def verifie_cbs(canal_cbs):
    rng = np.random.default_rng(1)
    b = rng.integers(0, 2, 200_000).astype(np.uint8)
    for pe in (0.0, 0.05, 0.3):
        y = np.asarray(canal_cbs(b, pe, rng))
        assert y.shape == b.shape and set(np.unique(y)) <= {0, 1}
        t = np.mean(y != b)
        assert abs(t - pe) < 5 * np.sqrt(max(pe * (1 - pe), 1e-9) / len(b)) + 1e-9, f"taux d'inversion {t:.4f} ≠ pe = {pe}"
    _ok("canal binaire symétrique")


def verifie_pe_gauss(pe_gauss, seuil_opt_gauss):
    assert abs(pe_gauss(1, 0.8, 0.2, 0.0) - 0.1056497737) < 1e-6, "pe_gauss(A=1, σ=0.8, p1=0.2, s=0) incorrect"
    assert abs(pe_gauss(1, 0.5, 0.3, 0.2) - 0.0221780627) < 1e-6, "pe_gauss(A=1, σ=0.5, p1=0.3, s=0.2) incorrect"
    assert abs(seuil_opt_gauss(1, 0.8, 0.2) - 0.4436141956) < 1e-6, "seuil optimal incorrect (A=1, σ=0.8, p1=0.2)"
    assert abs(seuil_opt_gauss(1.5, 0.6, 0.35) - 0.0742847050) < 1e-6, "seuil optimal incorrect (A=1.5, σ=0.6, p1=0.35)"
    _ok("probabilité d'erreur et seuil optimal (gaussien)")


def verifie_pe_uniforme(pe_uniforme):
    assert abs(pe_uniforme(1, 1.5, 0.5, 0.0) - 1 / 6) < 1e-9, "pe_uniforme(A=1, a=1.5, p1=0.5, s=0) incorrect"
    assert abs(pe_uniforme(1, 2.0, 0.3, 0.4) - 0.21) < 1e-9, "pe_uniforme(A=1, a=2, p1=0.3, s=0.4) incorrect"
    assert abs(pe_uniforme(1, 0.8, 0.3, 0.1)) < 1e-12, "sans recouvrement (a < A) la probabilité d'erreur doit être nulle"
    _ok("probabilité d'erreur (bruit uniforme)")


def verifie_integrateur(integre_et_vide):
    r = np.arange(12, dtype=float)
    z = np.asarray(integre_et_vide(r, 4))
    assert np.allclose(z, [1.5, 5.5, 9.5]), "integre_et_vide doit rendre la MOYENNE de chaque bloc de L échantillons"
    _ok("intégrateur + échantillonneur")


def verifie_sigma(sigma_echantillon):
    assert abs(sigma_echantillon(0, 1, 1) - np.sqrt(0.5)) < 1e-9, "Eb/N0 = 0 dB, L = 1 : σ² doit valoir N0/2 = 0.5"
    assert abs(sigma_echantillon(3, 1, 8) - np.sqrt(8 * 10 ** (-0.3) / 2)) < 1e-9, "dépendance en L incorrecte"
    _ok("variance du bruit par échantillon")


def verifie_hamming(G, H, hamming_code, hamming_decode):
    G, H = np.asarray(G), np.asarray(H)
    assert G.shape == (4, 7) and H.shape == (3, 7), "G doit être 4×7 et H 3×7 (à saisir après la question Q4.1)"
    G, H = G.astype(int) % 2, H.astype(int) % 2
    assert not (G @ H.T % 2).any(), "G·Hᵀ doit être nulle"
    assert (G[:, :4] == np.eye(4)).all(), "code systématique attendu : G = [I4 | P]"
    msgs = np.array([[(k >> i) & 1 for i in range(4)] for k in range(16)], dtype=np.uint8)
    C = np.asarray(hamming_code(msgs.ravel())).reshape(16, 7)
    assert (C == msgs @ G % 2).all(), "hamming_code ne correspond pas à G"
    dmin = min(int(c.sum()) for c in C if c.any())
    assert dmin == 3, f"distance minimale = {dmin}, attendu 3"
    for k in range(16):
        assert (np.asarray(hamming_decode(C[k])) == msgs[k]).all(), "décodage sans erreur incorrect"
        for j in range(7):
            e = C[k].copy()
            e[j] ^= 1
            assert (np.asarray(hamming_decode(e)) == msgs[k]).all(), f"erreur simple en position {j} non corrigée"
    _ok("code de Hamming (7,4) : G, H, codage, décodage par syndrome")


def verifie_registre(codeur_registre, hamming_code):
    for k in range(16):
        m = np.array([(k >> i) & 1 for i in range(4)], dtype=np.uint8)
        c1 = np.asarray(codeur_registre(m)).astype(int)
        c2 = np.asarray(hamming_code(m)).astype(int)
        assert (c1 == c2).all(), f"message {m} : registre {c1} ≠ matriciel {c2}"
    _ok("codeur à registre à décalage identique au codeur matriciel")


def verifie_qpsk(qpsk_mod, qpsk_demod):
    b = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    s = np.asarray(qpsk_mod(b))
    assert s.shape == (4,), "4 symboles attendus pour 8 bits"
    assert np.allclose(np.abs(s), 1), "les symboles doivent être d'énergie Es = 1"
    k = np.round((np.angle(s) / (np.pi / 4))).astype(int) % 8
    assert set(k) == {1, 3, 5, 7}, "symboles attendus : e^{jkπ/4}, k ∈ {1,3,5,7}"
    # Gray : deux symboles voisins (écart π/2) diffèrent d'un seul bit
    pairs = b.reshape(4, 2)
    for i in range(4):
        for j in range(4):
            if abs(abs(np.angle(s[i] / s[j])) - np.pi / 2) < 1e-6:
                assert (pairs[i] != pairs[j]).sum() == 1, "le mapping n'est pas un code de Gray"
    assert (np.asarray(qpsk_demod(s)) == b).all(), "qpsk_demod(qpsk_mod(b)) ≠ b"
    _ok("modulation / démodulation QPSK (Gray, Es = 1)")


def verifie_rrc(filtre_rrc):
    h = np.asarray(filtre_rrc(0.25, 4, 8))
    assert len(h) == 33, "longueur attendue span*L + 1"
    assert abs(np.sum(h**2) - 1) < 1e-9, "le filtre doit être normalisé à énergie unité"
    assert np.allclose(h, h[::-1]), "le filtre doit être symétrique"
    assert np.isfinite(h).all(), "valeurs non finies : traiter t = 0 et t = ±T/(4α)"
    assert abs(h[16] - 0.5342700667) < 1e-6 and abs(h[20] + 0.0321255002) < 1e-6, "valeurs de h incorrectes"
    rc = np.convolve(h, h)
    m = len(rc) // 2
    assert np.max(np.abs(rc[m + 4 :: 4])) < 0.02, "h * h ne vérifie pas (approximativement) le critère de Nyquist"
    _ok("filtre racine de cosinus surélevé")


def verifie_courbe(ebn0_dB, teb_sim, n_bits, teb_theo, nom="courbe"):
    """Vérifie qu'une courbe simulée est compatible avec la théorie."""
    teb_sim, teb_theo = np.asarray(teb_sim), np.asarray(teb_theo)
    n_bits = np.broadcast_to(n_bits, teb_sim.shape)
    for e, ts, n, tt in zip(ebn0_dB, teb_sim, n_bits, teb_theo):
        if tt * n < 50:
            print(f"   ({e} dB : moins de 50 erreurs attendues, point non vérifié — augmentez N)")
            continue
        lo, hi = intervalle_confiance(tt, n, z=4)
        assert lo <= ts <= hi, f"{nom} : à {e} dB, TEB simulé {ts:.3e} incompatible avec la théorie {tt:.3e}"
    _ok(f"{nom} compatible avec la théorie")


# ===================================================================== codage de source
def lire_texte(fichier="texte_fr.txt"):
    """Lit un fichier texte (UTF-8, ou latin-1 à défaut) pour la partie 2."""
    try:
        with open(fichier, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fichier, encoding="latin-1") as f:
            return f.read()


def verifie_entropie(entropie_empirique):
    assert abs(entropie_empirique("aabb") - 1) < 1e-12, "entropie_empirique('aabb') doit valoir 1 bit/symbole"
    assert abs(entropie_empirique("aaaa")) < 1e-12, "une suite constante a une entropie nulle"
    assert abs(entropie_empirique("abcd" * 25) - 2) < 1e-12, "4 symboles équiprobables : 2 bits/symbole"
    assert abs(entropie_empirique([0, 0, 0, 1]) - 0.8112781245) < 1e-9, "doit aussi accepter une liste de bits"
    _ok("entropie empirique")


def _est_prefixe(code):
    mots = sorted(code.values())
    return all(not mots[i + 1].startswith(mots[i]) for i in range(len(mots) - 1))


def verifie_huffman(huffman_code):
    for probas, Lattendue in [({"a": .4, "b": .2, "c": .2, "d": .1, "e": .1}, 2.2),
                              ({"a": .5, "b": .25, "c": .125, "d": .125}, 1.75),
                              ({"x": .9, "y": .1}, 1.0)]:
        code = huffman_code(probas)
        assert set(code) == set(probas), "le code doit contenir un mot pour chaque symbole"
        assert all(set(m) <= {"0", "1"} and len(m) > 0 for m in code.values()), "mots-codes : chaînes non vides de '0' et '1'"
        assert _est_prefixe(code), "le code n'est pas un code préfixe"
        kraft = sum(2.0 ** -len(m) for m in code.values())
        assert abs(kraft - 1) < 1e-12, f"somme de Kraft = {kraft} (un code de Huffman doit être complet)"
        L = sum(probas[s] * len(code[s]) for s in probas)
        assert abs(L - Lattendue) < 1e-9, f"longueur moyenne {L:.3f}, attendu {Lattendue}"
    _ok("construction du code de Huffman")


def verifie_codec_source(code_fn, decode_fn, message, nom="codage de source"):
    msg = message
    bits = code_fn(msg)
    assert set(bits) <= {"0", "1"}, "le codeur doit rendre une chaîne de '0' et de '1'"
    assert decode_fn(bits) == msg, "decode(code(message)) ≠ message"
    _ok(f"{nom} : décodage exact")


def verifie_lz78(lz78_code, lz78_decode):
    paires = lz78_code("ABAABABAABAB")
    attendu = [(0, "A"), (0, "B"), (1, "A"), (2, "A"), (4, "A"), (4, "B")]
    assert [tuple(p) for p in paires] == attendu, f"LZ78('ABAABABAABAB') = {paires}, attendu {attendu}"
    for s in ["ABAABABAABAB", "AAAAAAA", "le codage de source, le codage de canal", "x"]:
        assert lz78_decode(lz78_code(s)) == s, f"décodage LZ78 incorrect pour {s!r}"
    _ok("Lempel-Ziv 78 : analyse et décodage")
