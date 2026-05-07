from utils.data import generate_mallows
from utils.generate_csv import generate_csv_from_rankings
import numpy as np
import pyflagr.Linear as Linear
import pyflagr.Majoritarian as Majoritarian
import pyflagr.Kemeny as Kemeny
import pyflagr.RRA as RRA

config = {"n_voters": 100, "m_items": 7, "phi": 0.5}
permutation = list(range(1, config["m_items"] + 1))
config["ref_ranking"] = list(np.random.permutation(permutation))
rankings = generate_mallows(
    config["n_voters"], config["m_items"], config["ref_ranking"], config["phi"]
)

generate_csv_from_rankings(rankings, filename="data/rankings.csv")

borda = Linear.BordaCount()
copeland = Majoritarian.CopelandWinners()
condorcet = Majoritarian.CondorcetWinners()
robust_ra = RRA.RRA(exact=True)
kemeny = Kemeny.KemenyOptimal()
methods = [borda, copeland, condorcet, robust_ra, kemeny]


lists = "data/rankings.csv"
reference = [int(x) for x in config["ref_ranking"]]
results = []
for method in methods:
    df_out, _ = method.aggregate(input_file=lists)
    estimated = list(df_out["ItemID"])
    diff = np.abs(np.array(estimated) - np.array(reference)).mean()
    results.append((method.__class__.__name__, estimated, diff))
print("Reference:", reference)
for res in results:
    print(f"{res[0]}: {res[1]}, Error: {res[2]:.2f}")
