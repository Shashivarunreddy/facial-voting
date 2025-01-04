import csv
from collections import Counter
import os

def update_vote_summary():
    votes_file = "Votes.csv"
    summary_file = "VoteSummary.csv"

    # Check if Votes.csv exists
    if not os.path.isfile(votes_file):
        print(f"{votes_file} not found. Make sure the file exists and try again.")
        return

    # Read Votes.csv and count votes
    try:
        with open(votes_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  # Skip the header row if it exists
            
            votes = []
            for row in reader:
                if len(row) > 1:  # Ensure the row has at least two columns
                    votes.append(row[1])  # Extract the vote column

            if not votes:
                print("No valid votes found in the file.")
                return
            
            vote_count = Counter(votes)

            # Print the vote counts for debugging
            print(f"Vote counts: {dict(vote_count)}")

            # Update or create VoteSummary.csv
            with open(summary_file, "w", newline="") as summaryfile:
                writer = csv.writer(summaryfile)
                writer.writerow(["Party/Option", "Total Votes"])  # Write header
                for option, count in vote_count.items():
                    writer.writerow([option, count])
            
            print(f"Vote summary has been updated in {summary_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_vote_summary()
