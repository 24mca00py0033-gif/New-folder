import random
from collections import deque

from config import SPREAD_PROBABILITY, MAX_SPREAD_DEPTH


class NeutralAgent:

    def __init__(self, network):
        self.network = network

    def spread_claim(
        self,
        spread_prob: float | None = None,
        max_depth: int | None = None,
    ) -> dict:
        if spread_prob is None:
            spread_prob = SPREAD_PROBABILITY
        if max_depth is None:
            max_depth = MAX_SPREAD_DEPTH

        G = self.network.graph

        
        source_nodes = [n for n in G.nodes() if G.nodes[n]["status"] == "infected"]
        if not source_nodes:
            return {"success": False, "error": "No infected source node found"}

        src_node = source_nodes[0]
        claim_text = G.nodes[src_node].get("claim_text", "")

     
        queue = deque([(src_node, 0)])
        visited = {src_node}
        spread_path = [src_node]
        spread_per_step = []
        current_depth = 0
        step_count = 0
        edges_in_cascade = []

        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                break

            
            if depth > current_depth:
                spread_per_step.append(step_count)
                step_count = 0
                current_depth = depth

            neighbours = list(G.neighbors(node))
            random.shuffle(neighbours)

            for nb in neighbours:
                G.nodes[nb]["exposure_count"] += 1

                if nb in visited:
                    continue
                if depth + 1 > max_depth:
                    continue

                nb_role = G.nodes[nb]["role"]

            
                if nb_role in ("influencer", "fact_checker", "moderator"):
                   
                    visited.add(nb)
                    G.nodes[nb]["exposure_count"] += 1
                    continue

               
                if random.random() < spread_prob:
                    visited.add(nb)
                    spread_path.append(nb)
                    step_count += 1
                    edges_in_cascade.append((node, nb))

                    G.nodes[nb]["status"] = "infected"
                    G.nodes[nb]["shared"] = True
                    G.nodes[nb]["infection_time"] = depth + 1
                    G.nodes[nb]["claim_text"] = claim_text
                    queue.append((nb, depth + 1))

        
        if step_count:
            spread_per_step.append(step_count)

        
        infected_nodes = [n for n in G.nodes() if G.nodes[n]["status"] == "infected"]
        total_infected = len(infected_nodes)
        total_nodes = self.network.num_nodes

        return {
            "success": True,
            "source_node": src_node,
            "total_spread": total_infected,
            "total_nodes": total_nodes,
            "penetration_rate": round(total_infected / total_nodes * 100, 2),
            "spread_path": spread_path,
            "spread_per_step": spread_per_step,
            "max_depth": current_depth,
            "edges": edges_in_cascade,
            "exposed_nodes": len(visited),
            "normal_users_infected": sum(
                1 for n in infected_nodes if G.nodes[n]["role"] == "normal"
            ),
        }

    def get_spread_summary(self, spread_result: dict) -> str:
      
        sr = spread_result
        return (
            f"\n{'='*60}\n"
            f"  NEUTRAL AGENT — CLAIM SPREAD RESULTS\n"
            f"{'='*60}\n"
            f"Total Nodes Infected  : {sr.get('total_spread', 0)} / {sr.get('total_nodes', 0)}\n"
            f"Network Penetration   : {sr.get('penetration_rate', 0)}%\n"
            f"Max Spread Depth      : {sr.get('max_depth', 0)} hops\n"
            f"Nodes Exposed         : {sr.get('exposed_nodes', 0)}\n"
            f"Normal Users Infected : {sr.get('normal_users_infected', 0)}\n"
            f"{'─'*60}\n"
        )
