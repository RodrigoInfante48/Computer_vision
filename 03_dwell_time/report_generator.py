from datetime import datetime
from pathlib import Path

from dwell_analyzer import DwellAnalyzer


def generate_report(analyzer: DwellAnalyzer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = analyzer.get_all_stats()

    if df.empty:
        print("\n[Report] No tracking data recorded.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"dwell_report_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    total_people = len(df)
    avg_dwell = df["dwell_seconds"].mean()
    max_row = df.loc[df["dwell_seconds"].idxmax()]

    short = (df["dwell_seconds"] < 30).sum()
    medium = ((df["dwell_seconds"] >= 30) & (df["dwell_seconds"] <= 120)).sum()
    long_ = (df["dwell_seconds"] > 120).sum()

    print("\n" + "=" * 50)
    print("  DWELL TIME REPORT")
    print("=" * 50)
    print(f"  Total unique persons detected : {total_people}")
    print(f"  Average dwell time            : {avg_dwell:.1f}s")
    print(f"  Longest stay                  : ID {int(max_row['track_id'])} — {max_row['dwell_seconds']:.1f}s")
    print(f"\n  Time distribution:")
    print(f"    < 30s  (green)  : {short} person(s)")
    print(f"    30-120s (yellow): {medium} person(s)")
    print(f"    > 120s (red)    : {long_} person(s)")
    print("=" * 50)
    print(f"  CSV saved to: {csv_path}")
    print("=" * 50 + "\n")
