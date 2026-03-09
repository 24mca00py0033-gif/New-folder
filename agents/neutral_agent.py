"""
Neutral Agent (Normal User Behaviour)
======================================
Provides helpers for formatting the multi-cascade spread results
produced by SocialNetwork.run_multi_cascade().
The actual BFS propagation logic now lives in social_network.py so that
every role-based decision is made *inside* the graph during traversal.
"""


class NeutralAgent:
    """
    Represents the collective behaviour of normal users in the network.
    Normal users reshare content with a base probability without
    fact-checking.  This class is kept as a thin formatting helper.
    """

    def __init__(self, network):
        self.network = network

    def get_spread_summary(self, spread_result: dict) -> str:
        """Human-readable summary of a multi-cascade spread run."""
        sr = spread_result
        cascades = sr.get("cascade_results", [])
        header = f"""
{'='*60}
  MULTI-CASCADE SPREAD RESULTS
{'='*60}
Total Cascades        : {sr.get('num_cascades', 0)}
Total Nodes Reached   : {sr.get('total_reached', 0)} / {sr.get('total_nodes', 0)}
Network Penetration   : {sr.get('penetration_rate', 0)}%
Max Spread Depth      : {sr.get('max_depth_reached', 0)} hops
Avg Viral Coefficient : {sr.get('viral_coefficient', 0)}
Total Exposures       : {sr.get('total_exposures', 0)}
Nodes Warned          : {sr.get('total_warned', 0)}
Nodes Blocked         : {sr.get('total_blocked', 0)}
{'─'*60}
"""
        per_cascade = ""
        for c in cascades:
            per_cascade += (
                f"  Cascade {c['cascade_id']}  │ src=User_{c['source_node']:>4d}  "
                f"│ reached={c['total_reached']:>4d}  "
                f"│ depth={c['max_depth']:>2d}  "
                f"│ blocked={c['blocked_count']}  "
                f"│ warned={c['warned_count']}\n"
                f"{'':>16s}│ claim: {c['claim'][:70]}{'…' if len(c['claim'])>70 else ''}\n"
            )

        return header + per_cascade + "─" * 60
