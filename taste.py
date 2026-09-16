"""Drinks as vectors, and how a vector becomes input to the fly's taste neurons.

Three steps, all linear algebra except the receptor curves:

  1. drink vector   x ∈ R^7    what is in the glass       [sugar, alcohol, caffeine, hops, salt, CO2, pH]
  2. taste vector   t ∈ [0,1]^7 how strongly each taste lights up
                                [sweet, water, bitter, Ir94e, low salt, high salt, fizz]
  3. neuron input   u = M · (t ⊙ max_hz)   M: (taste neurons × 7) 0/1 matrix, one row per real GRN

Only taste neurons of organs that touch the drink get a row in M: legs (stepping in it),
labellum (tasting it) and pharynx (swallowing it). Wing taste bristles are left out.

Channel -> cell types follow the taste-connectome paper (bioRxiv 10.1101/2025.08.25.671814):
  sweet LB3b/LB3c (Gr64f), dorsal tpGRN (Gr5a), PhG1a-c (Gr64e), LgLG3 (Gr5a), LgLG4 (Gr64f), LgAG2 (Gr61a)
  water LB3a, PhG3/PhG4 (ppk28)          bitter LB1a-d, LgAG1 (Gr33a)        Ir94e LB1e
  low salt LB3b, LgLG4 (Ir56b)           high salt LB3d (Ir7c/ppk23/Ir47a)   fizz claw tpGRN (Ir56d)

Step 2's curves are ASSUMPTIONS (saturating Hill curves with plausible half-max points). Two
literature-motivated cross-talks: acid activates bitter and dampens sweet GRNs (Charlu et al. 2013),
and strong alcohol reads as bitter. The connectome has no dedicated alcohol or sour taste neurons.
"""
import numpy as np

INGREDIENTS = ["sugar_gL", "abv_pct", "caffeine_mgL", "ibu", "salt_gL", "co2_gL", "ph"]
CHANNELS = ["sweet", "water", "bitter", "ir94e", "low_salt", "high_salt", "fizz"]
CHANNEL_TYPES = {
    "sweet": ["LB3b", "LB3c", "dorsal_tpGRN", "PhG1a", "PhG1b", "PhG1c", "LgLG3", "LgLG4", "LgAG2"],
    "water": ["LB3a", "PhG3", "PhG4"],
    "bitter": ["LB1a", "LB1b", "LB1c", "LB1d", "LgAG1"],
    "ir94e": ["LB1e"],
    "low_salt": ["LB3b", "LgLG4"],
    "high_salt": ["LB3d"],
    "fizz": ["claw_tpGRN"],
}
MAX_HZ = np.array([200, 150, 150, 60, 80, 150, 120], dtype=float)  # firing at full activation, per channel
NEURON_CAP_HZ = 250.0
ORGAN_OF_SUBCLASS = {"leg bristle": "legs", "labellar bristle": "labellum", "taste peg": "labellum",
                     "pharyngeal sensillum": "pharynx"}
ORGANS = ("legs", "labellum", "pharynx")

#                         sugar  alc  caffeine  hops  salt  CO2   pH
DRINKS = {
    "💧 Νερό":             [0,    0,    0,     0,  0.02,  0,  7.0],
    "🫧 Σόδα":             [0,    0,    0,     0,  0.10,  6,  5.0],
    "🥤 Κόλα":             [106,  0,   96,     0,  0.10,  6,  2.5],
    "🍊 Χυμός πορτοκάλι":  [90,   0,    0,     0,  0.02,  0,  3.7],
    "🍋 Λεμονάδα":         [100,  0,    0,     0,  0.05,  5,  2.6],
    "⚡ Ενεργειακό ποτό":  [110,  0,  320,     0,  0.50,  5,  3.3],
    "🏃 Isotonic":         [60,   0,    0,     0,  1.20,  0,  3.0],
    "🧋 Φραπές γλυκός":    [50,   0,  500,     0,  0,     0,  5.0],
    "☕ Espresso":         [0,    0, 2100,     0,  0,     0,  5.2],
    "🍺 Μπίρα λάγκερ":     [3,    5.0,  0,    18,  0.05,  5,  4.3],
    "🍻 Μπίρα IPA":        [5,    6.5,  0,    60,  0.05, 4.8, 4.3],
    "🍷 Κόκκινο κρασί":    [2,   13.5,  0,     0,  0.02,  0,  3.5],
    "🍯 Μοσχάτο Σάμου":    [130, 15.0,  0,     0,  0.02,  0,  3.6],
    "🥂 Αφρώδες brut":     [8,   12.0,  0,     0,  0,    11,  3.2],
    "🥃 Τσίπουρο":         [0,   40.0,  0,     0,  0,     0,  7.0],
    "🫖 Kombucha":         [20,   0.5, 20,     0,  0,     3,  3.0],
}
DRINKS = {name: np.array(v, dtype=float) for name, v in DRINKS.items()}

# The user's cocktail menu. Their "bitterness" column (Campari, bitters, tonic quinine) sits in the ibu slot.
#                            sugar   alc  caffeine bitter salt  CO2   pH
COCKTAILS = {
    "🥃 Negroni":            [104,  21.6,  0,     90,   0,    0,   3.8],
    "🌿 Mojito":             [58,   11.0,  0,     10,   0,    3.0, 3.0],
    "🍹 Paloma":             [55,   11.0,  0,     30,   2.5,  3.0, 3.0],
    "🥃 Old Fashioned":      [51,   30.6,  0,     20,   0,    0,   4.5],
    "🍸 Pornstar Martini":   [119,  14.1,  0,      0,   0,    0,   3.2],
    "🥂 Aperol Spritz":      [75,    8.2,  0,     50,   0,    4.0, 3.5],
    "🍸 Cosmopolitan":       [52.5, 15.7,  0,      5,   0,    0,   2.8],
    "🍍 Piña Colada":        [89.5, 10.5,  0,      0,   0,    0,   4.0],
    "🧊 Gin & Tonic":        [58.7,  8.7,  0,     60,   0,    3.5, 2.8],
}
COCKTAILS = {name: np.array(v, dtype=float) for name, v in COCKTAILS.items()}
MENUS = {"classic": DRINKS, "cocktails": COCKTAILS}


def mix(parts):
    """Cocktail = weighted average of drink vectors. pH mixes through [H+], not linearly.

    parts: {drink name or vector: volume fraction}
    """
    total = sum(parts.values())
    vecs = [(DRINKS[k] if isinstance(k, str) else np.asarray(k, float), w / total) for k, w in parts.items()]
    x = sum(w * v for v, w in vecs)
    x[6] = -np.log10(sum(w * 10 ** -v[6] for v, w in vecs))
    return x


def hill(x, half, n=1.0):
    x = max(float(x), 0.0)
    return x ** n / (x ** n + half ** n)


def taste_vector(x):
    """Step 2: drink vector -> activation of each taste channel, 0..1."""
    sugar_gL, abv, caffeine_mgL, ibu, salt_gL, co2_gL, ph = x
    sugar_mM = sugar_gL / 342.3 * 1000      # sucrose-equivalent
    salt_mM = salt_gL / 58.44 * 1000
    ethanol_mM = abv * 171.3                # 1% v/v ethanol ~ 171 mM
    acid = min(max((4.0 - ph) / 2.0, 0.0), 1.0)   # 0 at pH >= 4, 1 at pH <= 2
    osmolarity = sugar_mM + 2 * salt_mM + ethanol_mM

    not_bitter = 1.0
    for part in (hill(caffeine_mgL / 194.2, 2.0), hill(ibu, 40), hill(abv, 15, 2), 0.5 * acid):
        not_bitter *= 1 - part
    return np.array([
        hill(sugar_mM, 30) * (1 - 0.5 * hill(abv, 20, 2)) * (1 - 0.4 * acid),  # sweet
        1 - hill(osmolarity, 150),                                            # water
        1 - not_bitter,                                                       # bitter
        hill(salt_mM, 50),                                                    # Ir94e
        hill(salt_mM, 20) * (1 - hill(salt_mM, 200, 2)),                      # low salt
        hill(salt_mM, 300, 2),                                                # high salt
        hill(co2_gL, 3),                                                      # fizz
    ])


def taste_matrix(meta, organs=ORGANS):
    """Step 3's M: rows = real taste neurons on the chosen organs, columns = channels (0/1).

    meta: brain_meta.parquet dataframe. Returns (neuron indices, M, organ of each row).
    """
    organ = meta["subclass"].map(ORGAN_OF_SUBCLASS)
    all_types = {t for ts in CHANNEL_TYPES.values() for t in ts}
    grn = meta[meta["type"].isin(all_types) & organ.isin(organs)]
    M = np.column_stack([grn["type"].isin(CHANNEL_TYPES[ch]).to_numpy() for ch in CHANNELS]).astype(float)
    return grn["idx"].to_numpy(), M, organ[grn.index].to_numpy()


def neuron_input(t, M):
    """Step 3: taste vector -> firing rate driven into each taste neuron (Hz)."""
    return np.minimum(M @ (t * MAX_HZ), NEURON_CAP_HZ)
