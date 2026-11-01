# Step 1 of the farm-episode pipeline: look inside a recording before extracting anything.
# Lists every topic (camera, GPS, odometry ...) with its message type, encoding and count.
#
#   python farm_inspect.py ~/datasets/citrusfarm_sample/data/06_14B_Jackal.fo.mcap
#
# Needs: pip install mcap

import sys

from mcap.reader import make_reader

path = sys.argv[1]
with open(path, "rb") as f:
    r = make_reader(f)
    s = r.get_summary()
    st = s.statistics
    dur = (st.message_end_time - st.message_start_time) / 1e9
    print(f"file: {path}\nduration: {dur:.1f} s, messages: {st.message_count}\n")
    print(f"{'topic':<45} {'message type':<40} {'encoding':<10} {'count':>7} {'Hz':>6}")
    for cid, ch in sorted(s.channels.items(), key=lambda kv: kv[1].topic):
        sc = s.schemas.get(ch.schema_id)
        n = st.channel_message_counts.get(cid, 0)
        print(f"{ch.topic:<45} {(sc.name if sc else '-'):<40} {ch.message_encoding:<10} {n:>7} {n / dur:>6.1f}")
    print("\nschema encodings:", sorted({sc.encoding for sc in s.schemas.values()}))
