import csv


def generate_csv_from_rankings(rankings, filename="rankings.csv"):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for voter_id, items in enumerate(rankings):
            for rank, item in enumerate(items, start=1):
                score = 1.0 - ((rank - 1) * (1.0 / len(items)))

                row = [
                    1,  # First column always 1
                    f"V-{voter_id}",  # Voter ID
                    f"{item}",  # Item ID
                    rank,  # Rank
                    f"{score:.5f}",  # Score
                    "Custom",  # Dataset name
                ]
                writer.writerow(row)
