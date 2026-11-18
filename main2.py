import numpy as np
import math
import matplotlib.pyplot as plt
from tp_outils import *
import heapq
import os
import zlib
import bz2
import lzma
from collections import Counter
from scipy.signal import fftconvolve, lfilter

#personal parameters (write the name as on the attendance list)
nom = "Marghich Bilal"
P = parametres_perso(nom)
print("Personal parameters:", P)

p1 = P["p1"]
p1_ext = P["p1_ext"]
pe = P["pe_cbs"]
A = P["A"]
a = P["a_unif"]
sigma = P["sigma"]
L = P["L"]
rng = np.random.default_rng(P["graine"])


fichier = "roman.txt" #Jules Verne, Le Tour du monde en quatre-vingts jours (Wikisource)

#figures are saved here for the report
os.makedirs("figures", exist_ok=True)
BLUE = "#2a78c5"
GREY = "#8a8a8a"
DARK = "#333333"
plt.rcParams["figure.max_open_warning"] = 0


#=========================================================== 2.1 Binary source
print("\n===== PART 2.1 =====")

#1)
#number of symboles
N = 100

#random source generator
def source(N, p1, rng):
    source = rng.choice([0, 1], N, p=[1 - p1, p1], replace=True)
    return source

symboles = source(N, p1, rng)
print("Generated array of symboles:", symboles)
verifie_source(source)

#2)
N_test = 100000
generator_test = source(N_test, p1, rng)

occurrence = np.count_nonzero(generator_test == 1) / N_test
low, high = intervalle_confiance(occurrence, N_test)
print("Percentage of 1s in the test generated array:", occurrence * 100, "%")
print("95% interval:", low, high, " expected p1 =", p1)

#3)
#calculate the entropy of the source
Entropy = H2(p1)
print("The entropy of the source is:", Entropy)

arr = np.linspace(0, 1, num=1000)
array = H2(arr)
plt.figure()
plt.plot(arr, array, color=BLUE, lw=2)
plt.plot([p1], [Entropy], "o", color=DARK, ms=8)
plt.annotate(f"$p_1$ = {p1} : H = {Entropy:.3f} bit", (p1, Entropy), xytext=(p1 + 0.04, Entropy - 0.08))
plt.xlabel("p")
plt.ylabel("H(p) (bit/symbole)")
plt.title("Entropie d'une source binaire sans mémoire")
plt.grid(alpha=0.3)
plt.savefig("figures/fig01_entropie.png", dpi=150, bbox_inches="tight")

#The information rate (1 bit emitted per second)
D = H2(p1) * 1
print("The information rate is:", D, "bit/s")


#=========================================================== 2.2 Text and Huffman
print("\n===== PART 2.2 =====")

#4)
seq = lire_texte(fichier)

def sequenceEntropy(seq):
    counter = Counter(seq)
    total = len(seq)
    probabilities = {}
    for letter in counter:
        probabilities[letter] = counter[letter] / total
    probability_values = list(probabilities.values())
    #calculate entropy
    Entropy = entropy(probability_values)
    return Entropy, set(counter), probabilities

#the function asked in the subject: only the entropy
def entropie_empirique(seq):
    return sequenceEntropy(seq)[0]

verifie_entropie(entropie_empirique)

H_text, unique_letters, probability_values = sequenceEntropy(seq)
#alphabet_size
alphabet_size = len(unique_letters)
# le nombre de bits d'un code de longueur fixe
bits_fixed = math.ceil(math.log2(alphabet_size))
#Redundancy
R_text = 1 - H_text / 8
print("Number of characters:", len(seq))
print("Alphabet size:", alphabet_size)
print("Entropy:", H_text, "bit/character")
print("Fixed length code:", bits_fixed, "bits")
print("Redundancy against 8 bits:", R_text * 100, "%")

#tracage des caracteres les plus frequents
top = sorted(probability_values.items(), key=lambda x: x[1], reverse=True)[:20]
chars = [repr(c) for c, _ in top]   # repr makes ' ' and '\n' visible
freqs = [q for _, q in top]
print("Most frequent:", top[:6])

plt.figure(figsize=(8, 4))
plt.bar(chars, freqs, color=BLUE)
plt.xlabel("Caractère")
plt.ylabel("Fréquence")
plt.title("Les 20 caractères les plus fréquents du texte")
plt.grid(axis="y", alpha=0.3)
plt.savefig("figures/fig02_frequences.png", dpi=150, bbox_inches="tight")

#5)
#Huffman code
#merging by hand of the test probabilities
heap = [0.4, 0.2, 0.2, 0.1, 0.1]
heapq.heapify(heap)
while len(heap) > 1:
    x = heapq.heappop(heap)
    y = heapq.heappop(heap)
    heapq.heappush(heap, x + y)
    print(heap)

def huffman_code(probs):
    heap = []
    #counter to avoid comparing dicts
    t = 0
    for symbol, q in probs.items():
        heap.append((q, t, {symbol: ""}))
        t += 1
    heapq.heapify(heap)
    #only one symbol: it still needs one bit
    if len(heap) == 1:
        return {symbol: "0" for symbol in probs}
    while len(heap) > 1:
        q0, t0, codes0 = heapq.heappop(heap)
        q1, t1, codes1 = heapq.heappop(heap)
        #left branch adds a 0, right branch adds a 1
        codes = {}
        for symbol in codes0:
            codes[symbol] = "0" + codes0[symbol]
        for symbol in codes1:
            codes[symbol] = "1" + codes1[symbol]
        heapq.heappush(heap, (q0 + q1, t, codes))
        t += 1
    return heap[0][2]

verifie_huffman(huffman_code)

def longueur_moyenne(probs, code):
    return sum(probs[s] * len(code[s]) for s in probs)

test_probs = {"a": .4, "b": .2, "c": .2, "d": .1, "e": .1}
test_code = huffman_code(test_probs)
print("Test code:", test_code)
print("Test mean length:", longueur_moyenne(test_probs, test_code))
print("Test Kraft sum:", sum(2.0 ** -len(m) for m in test_code.values()))

#6)
code_text = huffman_code(probability_values)
L_mean = longueur_moyenne(probability_values, code_text)
efficiency = H_text / L_mean
print("Mean length:", L_mean, "bit/character")
print("Efficiency:", efficiency * 100, "%")
print("H <= L < H+1 :", H_text <= L_mean < H_text + 1)
lengths = [len(m) for m in code_text.values()]
print("Shortest and longest word:", min(lengths), max(lengths))

def huffman_encode(seq, code):
    return "".join(code[c] for c in seq)

def huffman_decode(bits, code):
    #we read bit by bit until the word is in the table (prefix code)
    inverse = {m: s for s, m in code.items()}
    message = []
    word = ""
    for b in bits:
        word += b
        if word in inverse:
            message.append(inverse[word])
            word = ""
    return "".join(message)

bits_text = huffman_encode(seq, code_text)
verifie_codec_source(lambda m: huffman_encode(m, code_text), lambda b: huffman_decode(b, code_text), seq, "Huffman texte")
print("Size before:", 8 * len(seq), "bits, after:", len(bits_text), "bits")
print("Compression rate:", 8 * len(seq) / len(bits_text), " gain:", (1 - len(bits_text) / (8 * len(seq))) * 100, "%")

#7)
#table = for each symbol: its UTF-8 bits + 8 bits for the length + the word
table_bits = sum(8 * len(s.encode("utf-8")) + 8 + len(m) for s, m in code_text.items())
print("Table cost:", table_bits, "bits =", table_bits / 8, "bytes")
print("Table share for this text:", table_bits / (table_bits + len(bits_text)) * 100, "%")
#length where the table is 1% of the compressed message
print("Length where table = 1% of the message:", table_bits / (0.01 * L_mean), "characters")


#=========================================================== 2.3 Source extension
print("\n===== PART 2.3 =====")

#8)
H_ext = H2(p1_ext)
print("p1_ext =", p1_ext, " H =", H_ext)
N_bits = 100000
bits_ext = source(N_bits, p1_ext, rng)
K = np.arange(1, 11)
L_theory = []
L_measure = []
for k in K:
    #probability of every block of k bits
    probs = {}
    for block in range(2 ** k):
        w = bin(block).count("1")
        probs[block] = p1_ext ** w * (1 - p1_ext) ** (k - w)
    code = huffman_code(probs)
    L_theory.append(longueur_moyenne(probs, code) / k)
    #real coding of 10^5 bits cut in blocks of k bits
    n_blocks = N_bits // k
    blocks = bits_ext[:n_blocks * k].reshape(-1, k)
    values = blocks @ (2 ** np.arange(k))
    length_table = np.array([len(code[b]) for b in range(2 ** k)])
    L_measure.append(length_table[values].sum() / (n_blocks * k))
    print(k, "theory:", round(L_theory[-1], 4), " measure:", round(L_measure[-1], 4), " table:", 2 ** k, "words")

plt.figure()
plt.plot(K, L_theory, "o-", color=BLUE, lw=2, label=r"$\bar{L}_k/k$ théorique")
plt.plot(K, L_measure, "x", color=DARK, ms=8, label="mesuré sur $10^5$ bits")
plt.plot(K, H_ext + 1 / K, "--", color=GREY, label="borne H + 1/k")
plt.axhline(H_ext, color=DARK, ls=":", label=f"entropie H = {H_ext:.3f}")
plt.xlabel("Taille k des blocs")
plt.ylabel("bit par bit de source")
plt.title(f"Extension de la source, $p_{{1,ext}}$ = {p1_ext}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/fig03_extension.png", dpi=150, bbox_inches="tight")


#=========================================================== 2.4 Memory and compressors
print("\n===== PART 2.4 =====")

#9)
def entropie_paires(seq):
    #consecutive pairs, sliding window
    pairs = [seq[i:i + 2] for i in range(len(seq) - 1)]
    return entropie_empirique(pairs) / 2

def compare_compresseurs(seq):
    n = len(seq)
    data = seq.encode("utf-8")
    H0, letters, probs = sequenceEntropy(seq)
    code = huffman_code(probs)
    results = {}
    results["Longueur fixe"] = math.ceil(math.log2(len(letters)))
    results["Entropie ordre 0"] = H0
    results["Huffman"] = longueur_moyenne(probs, code)
    results["H(X1X2)/2"] = entropie_paires(seq)
    results["zlib"] = 8 * len(zlib.compress(data, 9)) / n
    results["bz2"] = 8 * len(bz2.compress(data, 9)) / n
    results["lzma"] = 8 * len(lzma.compress(data, preset=9)) / n
    return results

#first 5000 characters
N_short = min(5000, len(seq) // 2)
compare_long = compare_compresseurs(seq)
compare_short = compare_compresseurs(seq[:N_short])
print("Whole text,", len(seq), "characters:", compare_long)
print("First", N_short, "characters:", compare_short)

names = list(compare_long.keys())
pos = np.arange(len(names))
plt.figure(figsize=(8, 4))
plt.barh(pos - 0.2, [compare_long[n] for n in names], 0.38, color=BLUE, label=f"Texte entier ({len(seq)} car.)")
plt.barh(pos + 0.2, [compare_short[n] for n in names], 0.38, color=GREY, label=f"{N_short} premiers caractères")
plt.yticks(pos, names)
plt.gca().invert_yaxis()
plt.xlabel("bits par caractère")
plt.title("Comparaison du codage de source et des compresseurs")
plt.legend()
plt.grid(axis="x", alpha=0.3)
plt.savefig("figures/fig04_compresseurs.png", dpi=150, bbox_inches="tight")

#10)
bits_array = np.array(list(bits_text)).astype(np.uint8)
print("Proportion of 1 after Huffman:", bits_array.mean())
print("Entropy of pairs of bits, per bit:", entropie_paires(bits_text))


#=========================================================== 2.5 Lempel-Ziv
print("\n===== PART 2.5 =====")

#11)
def lz78_code(seq):
    dictionary = {"": 0}
    pairs = []
    phrase = ""
    for c in seq:
        if phrase + c in dictionary:
            phrase += c
        else:
            pairs.append((dictionary[phrase], c))
            dictionary[phrase + c] = len(dictionary)
            phrase = ""
    #the end of the message can be a phrase already known
    if phrase != "":
        pairs.append((dictionary[phrase[:-1]], phrase[-1]))
    return pairs

def lz78_decode(pairs):
    phrases = [""]
    message = []
    for index, c in pairs:
        phrase = phrases[index] + c
        phrases.append(phrase)
        message.append(phrase)
    return "".join(message)

verifie_lz78(lz78_code, lz78_decode)
print(lz78_code("ABAABABAABAB"))

#12)
def lz78_bits(seq):
    pairs = lz78_code(seq)
    alphabet = len(set(seq))
    bits_symbol = math.ceil(math.log2(alphabet)) if alphabet > 1 else 1
    total = 0
    for i in range(1, len(pairs) + 1):
        total += math.ceil(math.log2(i)) + bits_symbol
    return total / len(seq)

print("LZ78 on the whole text:", lz78_bits(seq), "bit/character")
print("LZ78 on the first", N_short, "characters:", lz78_bits(seq[:N_short]))

sizes = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
lz_binary = []
for n in sizes:
    message = "".join(map(str, source(n, p1_ext, rng)))
    lz_binary.append(lz78_bits(message))
    print("LZ78 binary source, n =", n, ":", lz_binary[-1], "bit/bit")

plt.figure()
plt.semilogx(sizes, lz_binary, "o-", color=BLUE, lw=2, label="LZ78")
plt.axhline(H_ext, color=DARK, ls=":", label=f"entropie H = {H_ext:.3f}")
plt.xlabel("Longueur du message (bits)")
plt.ylabel("bit par bit de source")
plt.title(f"Convergence de LZ78, source binaire $p_{{1,ext}}$ = {p1_ext}")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.savefig("figures/fig05_lz78.png", dpi=150, bbox_inches="tight")


#=========================================================== 3 Basic chain
print("\n===== PART 3 =====")

#1)
def canal_cbs(b, pe, rng):
    #each bit is flipped with probability pe
    return b ^ (rng.random(len(b)) < pe)

verifie_cbs(canal_cbs)

#2)
N_sim = 1000000
bits = source(N_sim, p1, rng)
received = canal_cbs(bits, pe, rng)
teb_cbs, err = teb(bits, received)
low, high = intervalle_confiance(teb_cbs, N_sim)
print("CBS: pe =", pe, " TEB =", teb_cbs, " errors =", err, " IC95 =", low, high)

#3)
C_cbs = 1 - H2(pe)
R_code = 4 / 7
print("Capacity:", C_cbs, " R =", R_code, " reliable possible:", R_code < C_cbs)
#pe where the capacity falls to 4/7 (dichotomy)
lo, hi = 0.0, 0.5
for i in range(60):
    mid = (lo + hi) / 2
    if 1 - H2(mid) > R_code:
        lo = mid
    else:
        hi = mid
print("Capacity = 4/7 for pe =", lo)

pe_axis = np.linspace(0, 1, 1000)
plt.figure()
plt.plot(pe_axis, 1 - H2(pe_axis), color=BLUE, lw=2, label="C(pe) = 1 - H(pe)")
plt.axhline(R_code, color=GREY, ls="--", label="R = 4/7")
plt.plot([pe], [C_cbs], "o", color=DARK, ms=8, label=f"notre canal : pe = {pe}, C = {C_cbs:.3f}")
plt.xlabel("pe")
plt.ylabel("Capacité (bit/bit émis)")
plt.title("Capacité du canal binaire symétrique")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/fig06_capacite.png", dpi=150, bbox_inches="tight")

#4)
#densities for the uniform noise
r_axis = np.linspace(-A - a - 0.5, A + a + 0.5, 2000)
f0 = (np.abs(r_axis + A) <= a) / (2 * a)
f1 = (np.abs(r_axis - A) <= a) / (2 * a)
plt.figure()
plt.plot(r_axis, f0, color=GREY, lw=2, label="f(r|0)")
plt.plot(r_axis, f1, color=BLUE, lw=2, ls="--", label="f(r|1)")
plt.axvspan(A - a, a - A, color=GREY, alpha=0.15, label=f"recouvrement [{A - a:.2f} ; {a - A:.2f}]")
plt.xlabel("r")
plt.ylabel("densité")
plt.title(f"Vraisemblances du canal à bruit uniforme, A = {A}, a = {a}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/fig07_uniforme_densites.png", dpi=150, bbox_inches="tight")

def seuil_opt_uniforme(A, a, p1):
    #no overlap: every threshold between a-A and A-a is perfect
    if a <= A:
        return 0.0
    #overlap: we always decide the most probable bit in the overlap
    if p1 < 0.5:
        return a - A
    if p1 > 0.5:
        return A - a
    return 0.0 #equal: any threshold of the overlap

s_uni = seuil_opt_uniforme(A, a, p1)
print("Uniform: optimal threshold =", s_uni)

#5)
def pe_uniforme(A, a, p1, s):
    #P(n > s+A) for the bit 0, P(n < s-A) for the bit 1
    e0 = np.clip((a - (s + A)) / (2 * a), 0, 1)
    e1 = np.clip((a + (s - A)) / (2 * a), 0, 1)
    return (1 - p1) * e0 + p1 * e1

verifie_pe_uniforme(pe_uniforme)
print("Uniform: Pe(0) =", pe_uniforme(A, a, p1, 0.0), " Pe(s*) =", pe_uniforme(A, a, p1, s_uni))

#6)
def emetteur(b, A, L=1):
    #bit 0 -> -A, bit 1 -> +A, repeated L times
    return np.repeat(A * (2 * np.asarray(b, dtype=float) - 1), L)

def canal_uniforme(x, a, rng):
    return x + rng.uniform(-a, a, len(x))

def decision(z, s):
    return (z > s).astype(np.uint8)

x = emetteur(bits, A)
r = canal_uniforme(x, a, rng)
seuils = np.linspace(-1.6, 1.6, 33)
teb_uni = [teb(bits, decision(r, s))[0] for s in seuils]
s_fine = np.linspace(-1.6, 1.6, 400)

plt.figure()
plt.plot(s_fine, pe_uniforme(A, a, p1, s_fine), color=BLUE, lw=2, label="théorie")
plt.plot(seuils, teb_uni, "x", color=DARK, ms=7, label="simulation, $10^6$ bits")
plt.axvline(s_uni, color=GREY, ls="--", label=f"seuil optimal s* = {s_uni:.2f}")
plt.xlabel("Seuil s")
plt.ylabel("Probabilité d'erreur")
plt.title(f"Canal à bruit uniforme, $p_1$ = {p1}, a = {a}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/fig08_uniforme_teb.png", dpi=150, bbox_inches="tight")
i_best = int(np.argmin(teb_uni))
print("Uniform: best threshold of the series", seuils[i_best], "TEB", teb_uni[i_best])
print("Uniform: simulated TEB(0) =", teb(bits, decision(r, 0.0))[0], " TEB(s*) =", teb(bits, decision(r, s_uni))[0])

#7)
def pe_gauss(A, sigma, p1, s):
    return (1 - p1) * Q((s + A) / sigma) + p1 * Q((A - s) / sigma)

def seuil_opt_gauss(A, sigma, p1):
    return sigma ** 2 / (2 * A) * math.log((1 - p1) / p1)

verifie_pe_gauss(pe_gauss, seuil_opt_gauss)

#8)
def canal_gauss(x, sigma, rng):
    return x + sigma * rng.standard_normal(len(x))

s_gauss = seuil_opt_gauss(A, sigma, p1)
r = canal_gauss(x, sigma, rng)
seuils = np.linspace(-1.0, 2.0, 31)
teb_gau = [teb(bits, decision(r, s))[0] for s in seuils]
s_fine = np.linspace(-1.0, 2.0, 400)

plt.figure()
plt.plot(s_fine, pe_gauss(A, sigma, p1, s_fine), color=BLUE, lw=2, label="théorie")
plt.plot(seuils, teb_gau, "x", color=DARK, ms=7, label="simulation, $10^6$ bits")
plt.axvline(s_gauss, color=GREY, ls="--", label=f"seuil optimal s* = {s_gauss:.3f}")
plt.xlabel("Seuil s")
plt.ylabel("Probabilité d'erreur")
plt.title(f"Canal gaussien, $p_1$ = {p1}, $\\sigma$ = {sigma}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/fig09_gauss_teb.png", dpi=150, bbox_inches="tight")

t_0 = teb(bits, decision(r, 0.0))[0]
t_opt = teb(bits, decision(r, s_gauss))[0]
print("Gaussian: s* =", s_gauss)
print("Gaussian theory: Pe(0) =", float(pe_gauss(A, sigma, p1, 0.0)), " Pe(s*) =", float(pe_gauss(A, sigma, p1, s_gauss)))
print("Gaussian simulation: TEB(0) =", t_0, " TEB(s*) =", t_opt)
print("Gain:", (1 - t_opt / t_0) * 100, "%")
print("Best threshold in the series:", seuils[int(np.argmin(teb_gau))])

#9)
print("Compare: CBS", pe, " uniform", float(pe_uniforme(A, a, p1, s_uni)), " gaussian", float(pe_gauss(A, sigma, p1, s_gauss)))

#10)
def integre_et_vide(r, L):
    #mean of every block of L samples
    return r.reshape(-1, L).mean(axis=1)

def sigma_echantillon(ebn0_dB, A, L):
    Eb = A ** 2
    N0 = Eb / 10 ** (ebn0_dB / 10)
    return math.sqrt(N0 * L / 2)

verifie_integrateur(integre_et_vide)
verifie_sigma(sigma_echantillon)

#11)
#check of the variance after the integrator
s_test = sigma_echantillon(3, A, L)
noise = s_test * rng.standard_normal(L * 200000)
print("Noise variance after integrator:", integre_et_vide(noise, L).var(), " expected:", s_test ** 2 / L)

#12)
#simulation by packets
def simulation_chaine(ebn0_dB, N_total, L, rng, packet=200000):
    err_int = 0
    err_one = 0
    for i in range(N_total // packet):
        b = source(packet, 0.5, rng).astype(np.uint8)
        x = emetteur(b, A, L)
        r = canal_gauss(x, sigma_echantillon(ebn0_dB, A, L), rng)
        err_int += teb(b, decision(integre_et_vide(r, L), 0))[1]
        #only one sample in the middle of each bit
        err_one += teb(b, decision(r[L // 2::L], 0))[1]
    return err_int / N_total, err_one / N_total

EbN0 = np.arange(0, 10)
N_total = 4000000
teb_int = []
teb_one = []
for e in EbN0:
    t1, t2 = simulation_chaine(e, N_total, L, rng)
    teb_int.append(t1)
    teb_one.append(t2)
    print(e, "dB: integrator", t1, "  one sample", t2)

ratio = 10 ** (EbN0 / 10)
teb_theo = Q(np.sqrt(2 * ratio))
teb_theo_one = Q(np.sqrt(2 * ratio / L))
verifie_courbe(EbN0, teb_int, N_total, teb_theo, "integrateur")
verifie_courbe(EbN0, teb_one, N_total, teb_theo_one, "un echantillon")
print("Theory integrator:", teb_theo)
print("Loss of one sample:", 10 * math.log10(L), "dB")

fig, ax = figure_teb("TEB de la chaîne de base, bruit gaussien, L = " + str(L))
ax.plot(EbN0, teb_theo, color=BLUE, lw=2, label=r"théorie $Q(\sqrt{2E_b/N_0})$")
ax.plot(EbN0, teb_int, "x", color=DARK, ms=8, label="simulation avec intégrateur")
ax.plot(EbN0, teb_theo_one, color=GREY, lw=2, ls="--", label=r"théorie $Q(\sqrt{2E_b/(LN_0)})$")
ax.plot(EbN0, teb_one, "+", color=DARK, ms=9, label="simulation, un échantillon")
ax.legend()
plt.savefig("figures/fig10_teb_base.png", dpi=150, bbox_inches="tight")

#13)
#uniform noise per sample, then integration
z_uni = integre_et_vide(rng.uniform(-a, a, L * 500000), L)
var_uni = a ** 2 / 3 / L
print("Uniform after integrator: variance", z_uni.var(), " expected", var_uni)
z_axis = np.linspace(-1.2, 1.2, 400)
plt.figure()
plt.hist(z_uni, bins=80, density=True, color=GREY, alpha=0.6, label="simulation")
plt.plot(z_axis, np.exp(-z_axis ** 2 / (2 * var_uni)) / math.sqrt(2 * math.pi * var_uni), color=BLUE, lw=2, label="loi normale de même variance")
plt.xlabel("bruit après intégration")
plt.ylabel("densité")
plt.title(f"Moyenne de L = {L} bruits uniformes sur [-{a} ; {a}]")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/fig11_tcl.png", dpi=150, bbox_inches="tight")


#=========================================================== 4.2 Hamming code C(7,4)
print("\n===== PART 4.2 =====")

#1)
#polynomials are lists of bits, index = degree
g = [1, 1, 0, 1]

def reste_poly(degree):
    #remainder of x^degree by g(x)
    p = [0] * (degree + 1)
    p[degree] = 1
    for i in range(degree, 2, -1):
        if p[i] == 1:
            for j in range(4):
                p[i - 3 + j] ^= g[j]
    return p[:3]

P_mat = np.array([reste_poly(3 + i) for i in range(4)], dtype=np.uint8)
print("Remainders of x^3 m(x) for m0, m1, m2, m3:")
print(P_mat)
G = np.hstack((np.eye(4, dtype=np.uint8), P_mat))
H = np.hstack((P_mat.T, np.eye(3, dtype=np.uint8)))
print("G =")
print(G)
print("H =")
print(H)
print("G H^T =")
print(G @ H.T % 2)

#2)
print("Columns of H:", [H[:, j].tolist() for j in range(7)])
messages = np.array([[(k >> i) & 1 for i in range(4)] for k in range(16)], dtype=np.uint8)
words = messages @ G % 2
weights = words.sum(axis=1)
print("Weights of the 16 codewords:", sorted(Counter(weights.tolist()).items()))
print("Minimal distance:", weights[weights > 0].min())

#3)
#syndrome -> position of the error, read in the columns of H
weights_syndrome = np.array([1, 2, 4])
position = -np.ones(8, dtype=int)
for j in range(7):
    position[H[:, j] @ weights_syndrome] = j

def hamming_code(m):
    return (np.asarray(m, dtype=np.uint8).reshape(-1, 4) @ G % 2).astype(np.uint8).ravel()

def hamming_decode(r):
    r = np.asarray(r, dtype=np.uint8).reshape(-1, 7).copy()
    syndrome = (r @ H.T % 2) @ weights_syndrome
    #we flip the bit shown by the syndrome, in all the blocks at once
    blocks = np.nonzero(syndrome)[0]
    r[blocks, position[syndrome[blocks]]] ^= 1
    return r[:, :4].ravel()

verifie_hamming(G, H, hamming_code, hamming_decode)

#syndrome of an error at each position, for the report
for j in range(7):
    e = np.zeros(7, dtype=np.uint8)
    e[j] = 1
    print("Error at position", j, "-> syndrome", e @ H.T % 2)

#validation on a long random message, one error per word
m_test = source(40000, 0.5, rng).astype(np.uint8)
c_test = hamming_code(m_test)
errors = rng.integers(0, 7, len(c_test) // 7)
c_test[np.arange(len(errors)) * 7 + errors] ^= 1
print("One error per word, wrong bits after decoding:", np.count_nonzero(hamming_decode(c_test) != m_test))

#two errors in a word: the decoder corrects the wrong bit
c_two = hamming_code(np.array([1, 0, 1, 1], dtype=np.uint8))
print("Codeword of [1 0 1 1]:", c_two)
c_two[0] ^= 1
c_two[1] ^= 1
print("With two errors, decoded:", hamming_decode(c_two))

#5)
#shift register, clock by clock
def registre_etats(m):
    s0, s1, s2 = 0, 0, 0
    states = []
    #the message enters highest degree first: m3, m2, m1, m0
    for bit in [m[3], m[2], m[1], m[0]]:
        f = int(bit) ^ s2
        s2 = s1
        s1 = s0 ^ f
        s0 = f
        states.append((int(bit), f, s0, s1, s2))
    return states

def codeur_registre(m):
    states = registre_etats(m)
    r0, r1, r2 = states[-1][2:]
    return np.array([m[0], m[1], m[2], m[3], r0, r1, r2], dtype=np.uint8)

verifie_registre(codeur_registre, hamming_code)
print("Register table for m = [1,0,1,1]: (input, f, s0, s1, s2)")
for line in registre_etats([1, 0, 1, 1]):
    print(line)
print("Codeword:", codeur_registre([1, 0, 1, 1]), " matrix:", hamming_code(np.array([1, 0, 1, 1])))

#6)
#Meggitt decoder, clock by clock
def syndrome_registre(word):
    #division of r(x) by g(x): r6 enters first
    s0, s1, s2 = 0, 0, 0
    for bit in word[::-1]:
        f = s2
        s2 = s1
        s1 = s0 ^ f
        s0 = int(bit) ^ f
    return s0, s1, s2

def meggitt_decode(c):
    #position in the polynomial: c = [m0 m1 m2 m3 r0 r1 r2]
    word = [c[4], c[5], c[6], c[0], c[1], c[2], c[3]]
    word = [int(b) for b in word]
    s0, s1, s2 = syndrome_registre(word)
    #x^6 mod g(x) = 1 + x^2: the syndrome of an error on the bit about to go out
    for degree in range(6, -1, -1):
        if (s0, s1, s2) == (1, 0, 1):
            word[degree] ^= 1
            s0, s2 = 0, 0
        #multiplication by x
        f = s2
        s2 = s1
        s1 = s0 ^ f
        s0 = f
    return np.array(word[3:], dtype=np.uint8)

bad = 0
for k in range(16):
    c = words[k]
    if (meggitt_decode(c) != messages[k]).any():
        bad += 1
    for j in range(7):
        e = c.copy()
        e[j] ^= 1
        if (meggitt_decode(e) != messages[k]).any():
            bad += 1
print("Meggitt decoder, wrong cases over 16 x 8:", bad)

#=========================================================== 4.2.3 Performances
print("\n===== PART 4.2.3 =====")

#7)
#probability that a block of 4 information bits is right
ratio = 10 ** (EbN0 / 10)
p_unc = Q(np.sqrt(2 * ratio)) #uncoded bit, energy Eb
p_cod = Q(np.sqrt(2 * 4 / 7 * ratio)) #coded bit, energy Ec = 4/7 Eb
bloc_unc = (1 - p_unc) ** 4
bloc_cod = (1 - p_cod) ** 7 + 7 * p_cod * (1 - p_cod) ** 6
for e in [0, 3, 6, 9]:
    print(e, "dB: p =", p_unc[e], " p coded =", p_cod[e], " P(block ok) uncoded =", bloc_unc[e], " coded =", bloc_cod[e])

#8)
#approximation j/7
def pb_approx(p):
    return sum(j / 7 * math.comb(7, j) * p ** j * (1 - p) ** (7 - j) for j in range(2, 8))

#exact Pb: decode all 128 error patterns
patterns = np.array([[(k >> i) & 1 for i in range(7)] for k in range(128)], dtype=np.uint8)
decoded = hamming_decode(patterns.ravel()).reshape(128, 4)
weight_pattern = patterns.sum(axis=1)
wrong_info = np.zeros(8)
for j in range(8):
    wrong_info[j] = decoded[weight_pattern == j].sum()
print("Wrong information bits, summed over the patterns of weight j:", wrong_info)
print("Mean fraction of wrong information bits for j = 2:", wrong_info[2] / (4 * math.comb(7, 2)))

def pb_exact(p):
    return sum(wrong_info[j] / 4 * p ** j * (1 - p) ** (7 - j) for j in range(8))

#two errors: what the decoder does
c_two = hamming_code(np.array([1, 0, 1, 1], dtype=np.uint8))
r_two = c_two.copy()
r_two[[0, 1]] ^= 1
syndrome_two = (r_two @ H.T % 2) @ weights_syndrome
print("Two errors on positions 0 and 1: the decoder flips position", position[syndrome_two])

for e in [0, 3, 6, 9]:
    print(e, "dB: Pb approx =", pb_approx(p_cod[e]), " Pb exact =", pb_exact(p_cod[e]), " ratio =", pb_exact(p_cod[e]) / pb_approx(p_cod[e]))

#9)
#comparison at the same noise variance (rate of the code forgotten)
pb_same_var = pb_exact(p_unc)

fig, ax = figure_teb("Code de Hamming C(7,4) : probabilité d'erreur théorique")
ax.plot(EbN0, p_unc, color=GREY, lw=2, label=r"sans codage $Q(\sqrt{2E_b/N_0})$")
ax.plot(EbN0, pb_exact(p_cod), color=BLUE, lw=2, label="avec codage, exacte")
ax.plot(EbN0, pb_approx(p_cod), color=BLUE, lw=2, ls="--", label="avec codage, approximation du sujet")
ax.plot(EbN0, pb_same_var, color=DARK, lw=1.5, ls=":", label="avec codage, variance égale (trompeur)")
ax.set_ylim(1e-8, 1)
ax.legend()
plt.savefig("figures/fig14_hamming_theorie.png", dpi=150, bbox_inches="tight")


#=========================================================== 4.2.4 Insertion in the chain
print("\n===== PART 4.2.4 =====")

#simulate until 100 errors
def mesure_teb(une_simulation, ebn0_dB, rng, n_paquet, err_min=100, n_max=4000000):
    err = 0
    n = 0
    while err < err_min and n < n_max:
        e, nb = une_simulation(n_paquet, ebn0_dB, rng)
        err += e
        n += nb
    return err / n, n

#10)
def sim_base_hamming(n, ebn0_dB, rng, coded=True):
    b = source(n, 0.5, rng).astype(np.uint8) #n is a multiple of 4
    if coded:
        c = hamming_code(b)
        amp = A * math.sqrt(4 / 7) #energy Ec = 4/7 Eb per coded bit
    else:
        c = b
        amp = A
    x = emetteur(c, amp, L)
    #Eb = energy per information bit
    r = canal_gauss(x, sigma_echantillon(ebn0_dB, A, L), rng)
    c_hat = decision(integre_et_vide(r, L), 0)
    if coded:
        b_hat = hamming_decode(c_hat)
    else:
        b_hat = c_hat
    return teb(b, b_hat)[1], n

teb_h_unc = []
teb_h_cod = []
n_h_unc = []
n_h_cod = []
for e in EbN0:
    t, n = mesure_teb(lambda n, e, rng: sim_base_hamming(n, e, rng, False), e, rng, 140000)
    teb_h_unc.append(t)
    n_h_unc.append(n)
    t, n = mesure_teb(lambda n, e, rng: sim_base_hamming(n, e, rng, True), e, rng, 140000)
    teb_h_cod.append(t)
    n_h_cod.append(n)
    print(e, "dB: uncoded", teb_h_unc[-1], " coded", teb_h_cod[-1], " bits", n_h_cod[-1])
verifie_courbe(EbN0, teb_h_unc, n_h_unc, p_unc, "chaine sans codage")
verifie_courbe(EbN0, teb_h_cod, n_h_cod, pb_exact(p_cod), "chaine avec Hamming")

#Shannon limits
R_code = 4 / 7
limite_R = 10 * math.log10((2 ** (2 * R_code) - 1) / (2 * R_code))
limite_abs = 10 * math.log10(math.log(2))
print("Shannon limit for R = 4/7:", limite_R, "dB, absolute limit:", limite_abs, "dB")

EbN0_fine = np.linspace(-2, 10, 200)
ratio_fine = 10 ** (EbN0_fine / 10)
fig, ax = figure_teb("TEB avec et sans code de Hamming C(7,4), L = " + str(L))
ax.plot(EbN0_fine, Q(np.sqrt(2 * ratio_fine)), color=GREY, lw=2, label="sans codage, théorie")
ax.plot(EbN0, teb_h_unc, "x", color=DARK, ms=8, label="sans codage, simulation")
ax.plot(EbN0_fine, pb_exact(Q(np.sqrt(2 * 4 / 7 * ratio_fine))), color=BLUE, lw=2, label="avec codage, théorie")
ax.plot(EbN0, teb_h_cod, "o", color=BLUE, mfc="none", ms=8, label="avec codage, simulation")
ax.axvline(limite_R, color=DARK, ls="--", lw=1)
ax.axvline(limite_abs, color=DARK, ls=":", lw=1)
ax.text(limite_R + 0.15, 3e-7, f"limite de Shannon\nR = 4/7 : {limite_R:.2f} dB", fontsize=9)
ax.text(limite_abs + 0.15, 3e-5, f"limite absolue\nln 2 : {limite_abs:.2f} dB", fontsize=9)
ax.set_ylim(1e-7, 1)
ax.legend(loc="upper right", fontsize=9)
plt.savefig("figures/fig15_teb_hamming.png", dpi=150, bbox_inches="tight")

#11)
#Eb/N0 needed for a target TEB (the TEB decreases with Eb/N0: bisection)
def ebn0_pour_teb(f, cible):
    lo, hi = -5.0, 20.0
    for i in range(60):
        mid = (lo + hi) / 2
        if f(mid) > cible:
            lo = mid
        else:
            hi = mid
    return lo

f_unc = lambda d: float(Q(math.sqrt(2 * 10 ** (d / 10))))
f_cod = lambda d: float(pb_exact(Q(math.sqrt(2 * 4 / 7 * 10 ** (d / 10)))))
#crossing point of the two theoretical curves
lo, hi = -5.0, 15.0
for i in range(60):
    mid = (lo + hi) / 2
    if f_cod(mid) > f_unc(mid):
        lo = mid
    else:
        hi = mid
print("Coding becomes better above", lo, "dB, TEB there:", f_unc(lo))
d_unc = ebn0_pour_teb(f_unc, 1e-4)
d_cod = ebn0_pour_teb(f_cod, 1e-4)
print("TEB 1e-4: uncoded", d_unc, "dB, coded", d_cod, "dB, coding gain", d_unc - d_cod, "dB")
print("Gap to Shannon (R = 4/7):", d_cod - limite_R, "dB")
for cible in [1e-5, 1e-6]:
    print("TEB", cible, ": gain", ebn0_pour_teb(f_unc, cible) - ebn0_pour_teb(f_cod, cible), "dB")

#12)
print("Channel bit rate for 1 information bit per second:", 7 / 4, "bit/s")


#=========================================================== 5 QPSK
print("\n===== PART 5 =====")

#1) 2)
#Gray mapping: first bit on the I axis, second bit on the Q axis
def qpsk_mod(b):
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    return ((2 * b[:, 0] - 1) + 1j * (2 * b[:, 1] - 1)) / math.sqrt(2)

def qpsk_demod(z):
    #decision on the sign of each axis
    b = np.zeros((len(z), 2), dtype=np.uint8)
    b[:, 0] = np.real(z) > 0
    b[:, 1] = np.imag(z) > 0
    return b.ravel()

verifie_qpsk(qpsk_mod, qpsk_demod)
for pair in ["00", "01", "11", "10"]:
    s = qpsk_mod([int(pair[0]), int(pair[1])])[0]
    print(pair, "-> k =", int(round(np.angle(s) / (np.pi / 4))) % 8)

def sigma_qpsk(ebn0_dB, R=1):
    #noise for Es = 1
    Eb = 1 / (2 * R)
    N0 = Eb / 10 ** (ebn0_dB / 10)
    return math.sqrt(N0 / 2) #on each axis

def canal_complexe(s, sigma, rng):
    return s + sigma * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))

#3)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, e in zip(axes, [0, 5, 10]):
    b = source(6000, 0.5, rng).astype(np.uint8)
    z = canal_complexe(qpsk_mod(b), sigma_qpsk(e), rng)
    trace_constellation(ax, z, f"$E_b/N_0$ = {e} dB")
    ax.plot(qpsk_mod([0, 0, 0, 1, 1, 0, 1, 1]).real, qpsk_mod([0, 0, 0, 1, 1, 0, 1, 1]).imag, "o", color="#c0392b", ms=7)
    for pair, s in zip(["00", "01", "10", "11"], qpsk_mod([0, 0, 0, 1, 1, 0, 1, 1])):
        ax.annotate(pair, (s.real, s.imag), xytext=(s.real * 1.9, s.imag * 1.9), ha="center", color="#c0392b", fontsize=11)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
fig.suptitle("Constellations QPSK reçues (mapping de Gray, $E_s$ = 1)")
plt.savefig("figures/fig16_constellations_qpsk.png", dpi=150, bbox_inches="tight")

#4) 5)
#natural mapping: 00, 01, 10, 11 -> k = 1, 3, 5, 7
def qpsk_mod_naturel(b):
    b = np.asarray(b, dtype=int).reshape(-1, 2)
    k = 2 * (2 * b[:, 0] + b[:, 1]) + 1
    return np.exp(1j * k * np.pi / 4)

def qpsk_demod_naturel(z):
    #number of the quadrant: 0 for k=1, 1 for k=3, 2 for k=5, 3 for k=7
    v = np.floor(np.angle(z) / (np.pi / 2)).astype(int) % 4
    b = np.zeros((len(z), 2), dtype=np.uint8)
    b[:, 0] = (v >> 1) & 1
    b[:, 1] = v & 1
    return b.ravel()

b = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
print("Natural mapping check:", qpsk_demod_naturel(qpsk_mod_naturel(b)), np.round(np.angle(qpsk_mod_naturel(b)) / (np.pi / 4)) % 8)

def sim_qpsk(n, ebn0_dB, rng, naturel=False):
    b = source(n, 0.5, rng).astype(np.uint8)
    if naturel:
        z = canal_complexe(qpsk_mod_naturel(b), sigma_qpsk(ebn0_dB), rng)
        return teb(b, qpsk_demod_naturel(z))[1], n
    z = canal_complexe(qpsk_mod(b), sigma_qpsk(ebn0_dB), rng)
    return teb(b, qpsk_demod(z))[1], n

teb_gray = []
teb_nat = []
n_gray = []
n_nat = []
for e in EbN0:
    t, n = mesure_teb(lambda n, e, rng: sim_qpsk(n, e, rng), e, rng, 400000)
    teb_gray.append(t)
    n_gray.append(n)
    t, n = mesure_teb(lambda n, e, rng: sim_qpsk(n, e, rng, True), e, rng, 400000)
    teb_nat.append(t)
    n_nat.append(n)
    print(e, "dB: Gray", teb_gray[-1], " natural", teb_nat[-1], " ratio", teb_nat[-1] / teb_gray[-1])
teb_nat_theo = 1.5 * p_unc - p_unc ** 2
verifie_courbe(EbN0, teb_gray, n_gray, p_unc, "QPSK Gray")
verifie_courbe(EbN0, teb_nat, n_nat, teb_nat_theo, "QPSK naturel")

fig, ax = figure_teb("TEB de la QPSK : mapping de Gray et mapping naturel")
ax.plot(EbN0, p_unc, color=BLUE, lw=2, label=r"Gray et chaîne binaire (partie 3), théorie $Q(\sqrt{2E_b/N_0})$")
ax.plot(EbN0, teb_gray, "x", color=DARK, ms=8, label="QPSK Gray, simulation")
ax.plot(EbN0, teb_int, "o", color=BLUE, mfc="none", ms=9, label="chaîne binaire partie 3, simulation")
ax.plot(EbN0, teb_nat_theo, color=GREY, lw=2, ls="--", label=r"naturel, théorie $\frac{3}{2}p - p^2$")
ax.plot(EbN0, teb_nat, "+", color=DARK, ms=9, label="QPSK naturel, simulation")
ax.legend(loc="lower left", fontsize=9)
plt.savefig("figures/fig17_teb_qpsk.png", dpi=150, bbox_inches="tight")

#6)
#a codeword has 7 bits: 2 codewords = 14 bits = 7 symbols, no stuffing bit
def sim_qpsk_hamming(n, ebn0_dB, rng):
    b = source(n, 0.5, rng).astype(np.uint8) #n multiple of 8
    c = hamming_code(b)
    z = canal_complexe(qpsk_mod(c), sigma_qpsk(ebn0_dB, 4 / 7), rng)
    return teb(b, hamming_decode(qpsk_demod(z)))[1], n

teb_qh = []
n_qh = []
for e in EbN0:
    t, n = mesure_teb(sim_qpsk_hamming, e, rng, 400000)
    teb_qh.append(t)
    n_qh.append(n)
    print(e, "dB: QPSK + Hamming", teb_qh[-1], " part 4", teb_h_cod[e])
verifie_courbe(EbN0, teb_qh, n_qh, pb_exact(p_cod), "QPSK + Hamming")

fig, ax = figure_teb("TEB avec code de Hamming : QPSK et chaîne binaire de la partie 4")
ax.plot(EbN0, p_unc, color=GREY, lw=2, label="sans codage, théorie")
ax.plot(EbN0, pb_exact(p_cod), color=BLUE, lw=2, label="Hamming, théorie")
ax.plot(EbN0, teb_h_cod, "o", color=BLUE, mfc="none", ms=9, label="Hamming, chaîne binaire (partie 4)")
ax.plot(EbN0, teb_qh, "x", color=DARK, ms=8, label="Hamming + QPSK, simulation")
ax.legend(loc="lower left")
plt.savefig("figures/fig18_teb_qpsk_hamming.png", dpi=150, bbox_inches="tight")


#=========================================================== 6 Shaping filters
print("\n===== PART 6 =====")
alpha = P["alpha"]
span = 10

#filters of unit energy
def filtre_porte(L):
    return np.ones(L) / math.sqrt(L)

def filtre_rrc(alpha, L, span):
    #same order as verifie_rrc
    t = np.arange(-span * L // 2, span * L // 2 + 1) / L #t / T
    h = np.zeros(len(t))
    for i in range(len(t)):
        ti = t[i]
        if ti == 0:
            #limit in t = 0
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-9:
            #limit when denominator = 0
            h[i] = alpha / math.sqrt(2) * ((1 + 2 / np.pi) * math.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * math.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (4 * alpha / np.pi * math.cos(np.pi * (1 + alpha) * ti) + (1 - alpha) * np.sinc((1 - alpha) * ti)) / (1 - (4 * alpha * ti) ** 2)
    return h / math.sqrt(np.sum(h ** 2))

verifie_rrc(filtre_rrc)

#transmitter: one symbol every L samples, then filtering
def emission(s, h, L):
    x = np.zeros(len(s) * L, dtype=complex)
    x[::L] = s
    return fftconvolve(x, h)

#matched filter, delay len(h)-1
def reception(r, h, L, n_symb, decalage=0):
    y = fftconvolve(r, h[::-1])
    debut = len(h) - 1 + decalage
    return y, y[debut: debut + n_symb * L: L]

#bits -> TEB for the whole chain (QPSK Gray, optional Hamming, filters)
def sim_filtre(n, ebn0_dB, rng, h, coded=False, decalage=0, echo=None, egaliseur=False):
    b = source(n, 0.5, rng).astype(np.uint8)
    if coded:
        c = hamming_code(b)
        R = 4 / 7
    else:
        c = b
        R = 1
    s = qpsk_mod(c)
    x = emission(s, h, L)
    if echo is not None:
        x = canal_echo(x, echo[0], echo[1], L)
    #unit energy filter: the energy of a symbol over its samples is Es = 1
    r = canal_complexe(x, sigma_qpsk(ebn0_dB, R), rng)
    y, z = reception(r, h, L, len(s), decalage)
    if egaliseur:
        z = egalise(z, echo[0], echo[1])
    c_hat = qpsk_demod(z)
    if coded:
        b_hat = hamming_decode(c_hat)
    else:
        b_hat = c_hat
    return teb(b, b_hat)[1], n

#1)
h_porte = filtre_porte(L)
#porte filter = sqrt(L) x integrator
r_test = rng.standard_normal(10 * L)
y_test = fftconvolve(r_test, h_porte)[L - 1::L][:10]
print("Porte filter / integrator:", y_test[:3] / integre_et_vide(r_test, L)[:3], " sqrt(L) =", math.sqrt(L))
teb_porte = []
n_porte = []
for e in EbN0:
    t, n = mesure_teb(lambda n, e, rng: sim_filtre(n, e, rng, h_porte), e, rng, 200000)
    teb_porte.append(t)
    n_porte.append(n)
print("Porte filter TEB:", teb_porte)
verifie_courbe(EbN0, teb_porte, n_porte, p_unc, "filtre porte")

#2) 3)
h_rrc = filtre_rrc(alpha, L, span)
print("RRC alpha =", alpha, ": h(0) =", h_rrc[span * L // 2], " length", len(h_rrc))
rc = np.convolve(h_rrc, h_rrc)
m = len(rc) // 2
k_T = np.arange(-span, span + 1)
print("h at multiples of T:", np.round(h_rrc[::L], 4))
print("h*h at multiples of T:", np.round(rc[m % L::L], 5))
print("Max |h*h(kT)| for k != 0 (truncation):", np.max(np.abs(np.delete(rc[m % L::L], span))))
print("Max |h(kT)| for k != 0:", np.max(np.abs(np.delete(h_rrc[::L], span // 2))))

t_h = (np.arange(len(h_rrc)) - (len(h_rrc) - 1) / 2) / L
t_rc = (np.arange(len(rc)) - m) / L
fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
axes[0].plot(t_h, h_rrc, color=BLUE, lw=2)
axes[0].plot(t_h[::L], h_rrc[::L], "o", color=DARK, ms=5, label="valeurs aux instants kT")
axes[0].set_title(f"Racine de cosinus surélevé h, $\\alpha$ = {alpha}, span = {span}")
axes[0].set_ylabel("h(t)")
axes[0].legend()
axes[1].plot(t_rc, rc, color=BLUE, lw=2)
axes[1].plot(t_rc[m % L::L], rc[m % L::L], "o", color=DARK, ms=5, label="valeurs aux instants kT : nulles pour k ≠ 0")
axes[1].set_title("Cosinus surélevé h * h (émission + filtre adapté)")
axes[1].set_xlabel("t / T")
axes[1].set_ylabel("(h * h)(t)")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-span, span + 1, 2))
plt.savefig("figures/fig19_rrc.png", dpi=150, bbox_inches="tight")

#4)
#power spectral density: mean of |FFT|^2 over segments
def dsp(x, nfft=1024):
    seg = x[: len(x) // nfft * nfft].reshape(-1, nfft)
    S = np.mean(np.abs(np.fft.fft(seg, axis=1)) ** 2, axis=0)
    f = np.fft.fftfreq(nfft, d=1 / L) #in 1/T, sampling frequency L/T
    return np.fft.fftshift(f), np.fft.fftshift(S)

def bande_occupee(f, S, part=0.99):
    #smallest B such that [-B, B] holds part of the power
    order = np.argsort(np.abs(f))
    cumul = np.cumsum(S[order]) / np.sum(S)
    return np.abs(f[order])[np.searchsorted(cumul, part)]

s_dsp = qpsk_mod(source(400000, 0.5, rng).astype(np.uint8))
f, S_porte = dsp(emission(s_dsp, h_porte, L))
f, S_rrc = dsp(emission(s_dsp, h_rrc, L))
B_porte = bande_occupee(f, S_porte)
B_rrc = bande_occupee(f, S_rrc)
print("99% bandwidth: porte", B_porte, "/T, RRC", B_rrc, "/T, (1+alpha)/(2T) =", (1 + alpha) / 2)
print("90% bandwidth: porte", bande_occupee(f, S_porte, 0.9), " RRC", bande_occupee(f, S_rrc, 0.9))

plt.figure(figsize=(8, 4.5))
ref = S_rrc.max()
plt.plot(f, 10 * np.log10(S_porte / ref), color=GREY, lw=1.5, label=f"filtre porte, bande à 99 % : {B_porte:.2f}/T")
plt.plot(f, 10 * np.log10(S_rrc / ref), color=BLUE, lw=2, label=f"racine de cosinus surélevé, bande à 99 % : {B_rrc:.2f}/T")
plt.axvline((1 + alpha) / 2, color=DARK, ls="--", lw=1, label=f"(1 + $\\alpha$)/(2T) = {(1 + alpha) / 2:.3f}/T")
plt.axvline(-(1 + alpha) / 2, color=DARK, ls="--", lw=1)
plt.xlim(-4, 4)
plt.ylim(-70, 5)
plt.xlabel("f T")
plt.ylabel("DSP (dB, référence : maximum RRC)")
plt.title("Densité spectrale de puissance du signal QPSK émis")
plt.legend(loc="lower center", fontsize=9)
plt.grid(alpha=0.3)
plt.savefig("figures/fig20_dsp.png", dpi=150, bbox_inches="tight")

#5)
s_eye = qpsk_mod(source(4000, 0.5, rng).astype(np.uint8))
x_eye = emission(s_eye, h_rrc, L)
y_clean, z_clean = reception(x_eye, h_rrc, L, len(s_eye))
y_noise, z_noise = reception(canal_complexe(x_eye, sigma_qpsk(10), rng), h_rrc, L, len(s_eye))
debut = len(h_rrc) - 1 + 20 * L
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
trace_oeil(axes[0], y_clean, L, debut, titre="Diagramme de l'œil (voie I), sans bruit")
trace_oeil(axes[1], y_noise, L, debut, titre="Diagramme de l'œil (voie I), $E_b/N_0$ = 10 dB")
trace_constellation(axes[2], z_noise, "Constellation après filtre adapté, 10 dB")
for ax in axes[:2]:
    ax.axvline(0, color="#c0392b", lw=1)
    ax.set_ylabel("amplitude")
plt.savefig("figures/fig21_oeil_rrc.png", dpi=150, bbox_inches="tight")
print("Max distance of noiseless samples to the symbols:", np.max(np.abs(z_clean[span:-span] - s_eye[span:-span])))

teb_rrc = []
n_rrc = []
teb_rrc_h = []
n_rrc_h = []
for e in EbN0:
    t, n = mesure_teb(lambda n, e, rng: sim_filtre(n, e, rng, h_rrc), e, rng, 200000)
    teb_rrc.append(t)
    n_rrc.append(n)
    t, n = mesure_teb(lambda n, e, rng: sim_filtre(n, e, rng, h_rrc, True), e, rng, 200000, n_max=3000000)
    teb_rrc_h.append(t)
    n_rrc_h.append(n)
    print(e, "dB: RRC", teb_rrc[-1], " RRC + Hamming", teb_rrc_h[-1])
verifie_courbe(EbN0, teb_rrc, n_rrc, p_unc, "chaine RRC")
verifie_courbe(EbN0, teb_rrc_h, n_rrc_h, pb_exact(p_cod), "chaine RRC + Hamming")

fig, ax = figure_teb(f"TEB de la chaîne avec filtres de mise en forme, $\\alpha$ = {alpha}")
ax.plot(EbN0, p_unc, color=GREY, lw=2, label="sans codage, théorie")
ax.plot(EbN0, teb_gray, "o", color=GREY, mfc="none", ms=9, label="QPSK sans filtre (partie 5)")
ax.plot(EbN0, teb_porte, "+", color=DARK, ms=9, label="QPSK + filtre porte")
ax.plot(EbN0, teb_rrc, "x", color=DARK, ms=8, label="QPSK + RRC")
ax.plot(EbN0, pb_exact(p_cod), color=BLUE, lw=2, label="Hamming, théorie")
ax.plot(EbN0, teb_rrc_h, "s", color=BLUE, mfc="none", ms=7, label="Hamming + QPSK + RRC")
ax.legend(loc="lower left", fontsize=9)
plt.savefig("figures/fig22_teb_rrc.png", dpi=150, bbox_inches="tight")

#6)
#sampling T/4 too late
decalage = L // 4
y_shift, z_shift = reception(canal_complexe(x_eye, sigma_qpsk(10), rng), h_rrc, L, len(s_eye), decalage)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, a_test in enumerate([0.22, 1.0]):
    h_test = filtre_rrc(a_test, L, span)
    x_test = emission(s_eye, h_test, L)
    y_test, z0 = reception(x_test, h_test, L, len(s_eye))
    y_n, z_t = reception(canal_complexe(x_test, sigma_qpsk(10), rng), h_test, L, len(s_eye), decalage)
    trace_oeil(axes[row, 0], y_test, L, len(h_test) - 1 + 20 * L, titre=f"Œil sans bruit, $\\alpha$ = {a_test}")
    axes[row, 0].axvline(0, color="#c0392b", lw=1, label="instant optimal")
    axes[row, 0].axvline(0.25, color="#c0392b", lw=1, ls="--", label="décalage T/4")
    axes[row, 0].legend(loc="lower right", fontsize=8)
    _, z_c = reception(x_test, h_test, L, len(s_eye), decalage)
    trace_constellation(axes[row, 1], z_c[span:-span], f"Sans bruit, décalage T/4, $\\alpha$ = {a_test}")
    trace_constellation(axes[row, 2], z_t[span:-span], f"10 dB, décalage T/4, $\\alpha$ = {a_test}")
    print("alpha", a_test, ": spread of the I samples at T/4 (no noise): min", np.min(np.abs(z_c[span:-span].real)), " max", np.max(np.abs(z_c[span:-span].real)))
plt.tight_layout()
plt.savefig("figures/fig23_decalage.png", dpi=150, bbox_inches="tight")

teb_dec = {}
for a_test in [0.22, alpha, 1.0]:
    h_test = filtre_rrc(a_test, L, span)
    teb_dec[a_test] = []
    for e in EbN0:
        t, n = mesure_teb(lambda n, e, rng: sim_filtre(n, e, rng, h_test, decalage=decalage), e, rng, 200000, n_max=2000000)
        teb_dec[a_test].append(t)
    print("alpha", a_test, "T/4 offset:", teb_dec[a_test])
    #loss at 9 dB
    print("   at 9 dB, loss in dB:", 9 - ebn0_pour_teb(f_unc, teb_dec[a_test][9]))

fig, ax = figure_teb("TEB avec un échantillonnage décalé de T/4")
ax.plot(EbN0, p_unc, color=GREY, lw=2, label="instant optimal, théorie")
shades = {0.22: 1.0, alpha: 0.65, 1.0: 0.35}
for a_test in [0.22, alpha, 1.0]:
    ax.plot(EbN0, teb_dec[a_test], "o-", color=BLUE, alpha=shades[a_test], lw=2, label=f"décalage T/4, $\\alpha$ = {a_test}")
ax.legend(loc="lower left")
plt.savefig("figures/fig24_teb_decalage.png", dpi=150, bbox_inches="tight")


#=========================================================== 7 Two-path channel
print("\n===== PART 7 =====")
beta = P["beta"]
retard = P["retard_trajet"]

#hc(t) = delta(t) + beta delta(t - tau), tau = retard symbols
def canal_echo(x, beta, retard, L):
    y = x.copy()
    d = retard * L
    y[d:] += beta * x[:-d]
    return y

#zero forcing equalizer
def egalise(z, beta, retard):
    den = np.zeros(retard + 1)
    den[0] = 1
    den[retard] = beta
    return lfilter([1], den, z)

#1)
x_echo = canal_echo(x_eye, beta, retard, L)
y_e, z_e = reception(x_echo, h_rrc, L, len(s_eye))
y_en, z_en = reception(canal_complexe(x_echo, sigma_qpsk(10), rng), h_rrc, L, len(s_eye))
niveaux = np.abs(z_e[span:-span].real)
print("Echo: |I| levels without noise:", niveaux[niveaux < 0.7].mean(), niveaux[niveaux > 0.7].mean(), " expected", (1 - beta) / math.sqrt(2), (1 + beta) / math.sqrt(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
trace_oeil(axes[0], y_e, L, len(h_rrc) - 1 + 20 * L, titre=f"Œil avec écho ($\\beta$ = {beta}, $\\tau$ = {retard}T), sans bruit")
trace_constellation(axes[1], z_e[span:-span], "Constellation avec écho, sans bruit")
trace_constellation(axes[2], egalise(z_en, beta, retard)[span:-span], "Après égaliseur, 10 dB")
axes[0].axvline(0, color="#c0392b", lw=1)
plt.savefig("figures/fig25_echo.png", dpi=150, bbox_inches="tight")

teb_echo = []
teb_egal = []
n_echo = []
n_egal = []
for e in EbN0:
    t, n = mesure_teb(lambda n, e, rng: sim_filtre(n, e, rng, h_rrc, echo=(beta, retard)), e, rng, 200000, n_max=3000000)
    teb_echo.append(t)
    n_echo.append(n)
    t, n = mesure_teb(lambda n, e, rng: sim_filtre(n, e, rng, h_rrc, echo=(beta, retard), egaliseur=True), e, rng, 200000, n_max=3000000)
    teb_egal.append(t)
    n_egal.append(n)
    print(e, "dB: echo", teb_echo[-1], " equalized", teb_egal[-1])
teb_echo_theo = 0.5 * (Q(np.sqrt(2 * ratio) * (1 + beta)) + Q(np.sqrt(2 * ratio) * (1 - beta)))
teb_egal_theo = Q(np.sqrt(2 * ratio * (1 - beta ** 2)))
verifie_courbe(EbN0, teb_echo, n_echo, teb_echo_theo, "canal a echo")
verifie_courbe(EbN0, teb_egal, n_egal, teb_egal_theo, "canal a echo egalise")
print("Loss at 1e-4: echo", ebn0_pour_teb(lambda d: float(0.5 * (Q(math.sqrt(2 * 10 ** (d / 10)) * (1 + beta)) + Q(math.sqrt(2 * 10 ** (d / 10)) * (1 - beta)))), 1e-4) - d_unc,
      "dB, equalized", -10 * math.log10(1 - beta ** 2), "dB")

fig, ax = figure_teb(f"TEB sur le canal à deux trajets, $\\beta$ = {beta}, $\\tau$ = {retard}T")
ax.plot(EbN0, p_unc, color=GREY, lw=2, label="canal idéal, théorie")
ax.plot(EbN0, teb_echo_theo, color=DARK, lw=2, ls="--", label="écho sans égaliseur, théorie")
ax.plot(EbN0, teb_echo, "x", color=DARK, ms=8, label="écho sans égaliseur, simulation")
ax.plot(EbN0, teb_egal_theo, color=BLUE, lw=2, label=r"égaliseur, théorie $Q(\sqrt{2(1-\beta^2)E_b/N_0})$")
ax.plot(EbN0, teb_egal, "o", color=BLUE, mfc="none", ms=8, label="égaliseur, simulation")
ax.legend(loc="lower left", fontsize=9)
plt.savefig("figures/fig26_teb_echo.png", dpi=150, bbox_inches="tight")

#3)
#send the text through the chain
texte = seq[:1200]
octets = np.frombuffer(texte.encode("utf-8"), dtype=np.uint8)
bits_texte = np.unpackbits(octets) #8 bits per byte: multiple of 8

def transmet_texte(bits_texte, ebn0_dB, coded, rng):
    if coded:
        c = hamming_code(bits_texte)
        R = 4 / 7
    else:
        c = bits_texte
        R = 1
    s = qpsk_mod(c)
    x = canal_echo(emission(s, h_rrc, L), beta, retard, L)
    y, z = reception(canal_complexe(x, sigma_qpsk(ebn0_dB, R), rng), h_rrc, L, len(s))
    c_hat = qpsk_demod(egalise(z, beta, retard))
    b_hat = hamming_decode(c_hat) if coded else c_hat
    octets_recus = np.packbits(b_hat)
    recu = octets_recus.tobytes().decode("utf-8", errors="replace")
    return recu, teb(bits_texte, b_hat)[0], octets_recus

print("Sent:", texte[:160].replace("\n", " "))
for e in [3, 6]:
    for coded in [False, True]:
        recu, t, octets_recus = transmet_texte(bits_texte, e, coded, rng)
        #count wrong bytes
        faux = np.count_nonzero(octets_recus != octets)
        print(e, "dB, Hamming" if coded else "dB, no code", ": TEB", t, " wrong bytes", faux, "/", len(octets))
        print("   ", recu[:160].replace("\n", " "))


#=========================================================== 8 Conclusion
print("\n===== PART 8 =====")
print("Eb/N0 for TEB 1e-4: uncoded", d_unc, " Hamming", d_cod, " Shannon R=4/7", limite_R, " absolute", limite_abs)
print("Bandwidth for 1 bit/s: binary porte ~", B_porte, " QPSK porte", B_porte / 2, " QPSK RRC", B_rrc / 2, " with Hamming", B_rrc / 2 * 7 / 4)

plt.show()
