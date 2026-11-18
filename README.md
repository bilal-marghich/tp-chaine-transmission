# TP Chaîne de transmission numérique (ENSIM 5A ASTRE)



Modélisation et simulation sous Python d'une chaîne de transmission numérique en bande de base : codage de source (Huffman, Lempel-Ziv), chaîne de base et détecteur optimal, code de Hamming C(7,4), modulation QPSK, filtres en racine de cosinus surélevé et canal à deux trajets avec égaliseur.

Le texte étudié dans la partie 2 est *Le Tour du monde en quatre-vingts jours* de Jules Verne (édition Wikisource, 419 408 caractères).

## Contenu du dépôt

| Fichier | Rôle |
| --- | --- |
| `main2.py` | Programme principal : toutes les parties du TP, de 2.1 à 8 |
| `tp_outils.py` | Fonctions fournies par l'enseignant (utilitaires, paramètres personnels, tests `verifie_*`). Ne pas modifier |
| `schema_registre.py` | Dessine le codeur à registre et le décodeur de Meggitt (partie 4.2.2) |
| `roman.txt` | Texte étudié dans la partie 2 (UTF-8) |
| `requirements.txt` | Bibliothèques Python nécessaires |
| `Compte rendu.pdf` | Compte rendu complet du TP, avec toutes les figures |
| `presentation.pptx` | Présentation de la vidéo (16 diapositives, script dans les notes) |
| `linktopresentation.txt` | Lien vers la présentation en ligne |

## Prérequis

- Python 3.9 ou plus récent
- Les bibliothèques NumPy, SciPy et Matplotlib :

```bash
pip install -r requirements.txt
```

## Lancer le TP

Depuis le dossier du dépôt :

```bash
python main2.py
```

- `tp_outils.py` et `roman.txt` doivent être dans le même dossier que `main2.py`.
- Les paramètres personnels sont calculés à partir du nom écrit en haut de `main2.py` (`nom = "Marghich Bilal"`). La graine est fixe : chaque exécution donne exactement les mêmes résultats.
- Chaque test réussi affiche une ligne `OK — …` (26 au total).
- Les figures sont enregistrées dans un dossier `figures/`, créé automatiquement.
- L'exécution complète dure environ 15 à 20 minutes, à cause des simulations de TEB sur plusieurs millions de bits.
- À la fin, `plt.show()` ouvre toutes les figures ; fermez les fenêtres pour terminer le programme.

Pour enregistrer la sortie console dans un fichier `resultats.txt`, sans ouvrir les fenêtres :

```bash
# Linux / macOS
MPLBACKEND=Agg python main2.py > resultats.txt
```

```powershell
# Windows PowerShell
$env:MPLBACKEND="Agg"; python main2.py > resultats.txt
```

Pour utiliser un autre texte dans la partie 2, changez la variable `fichier` en haut de `main2.py`.

## Schémas de la partie 4.2.2

Lancez d'abord `main2.py` (il crée le dossier `figures/`), puis :

```bash
python schema_registre.py
```

Les schémas sont enregistrés dans `figures/fig12_codeur_registre.png` et `figures/fig13_meggitt.png`.

## Compte rendu

Le compte rendu est le fichier `Compte rendu.pdf`. Ses résultats numériques et ses figures sont ceux produits par `main2.py`.

## Présentation

La présentation est le fichier `presentation.pptx`.
Le lien vers la vidéo de la présentation en ligne est dans `linktopresentation.txt`.
