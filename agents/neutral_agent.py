"""
Neutral Agent
Simulates average social media users who share content without verification.
Uses BFS traversal to model viral spread through the social network.
"""
import random
from collections import deque
from config import MAX_SPREAD_DEPTH, SPREAD_PROBABILITY


class NeutralAgent:
    """
    Simulates neutral social media users who reshare content without fact-checking.
    Implements BFS-based spread propagation through the network graph.
    """

    def __init__(self, network):
        """
        Args:
            network: SocialNetwork instance containing the graph
        """
        self.network = network
        self.graph = network.graph
        self.spread_probability = SPREAD_PROBABILITY
        self.max_depth = MAX_SPREAD_DEPTH

    def spread_claim(self, start_node, claim_text, max_depth=None, spread_prob=None):
        """
        Spread a claim through the network using BFS traversal.
        
        Args:
            start_node: Node ID where the claim originates
            claim_text: The claim text being spread
            max_depth: Maximum BFS depth (default from config)
            spread_prob: Probability each node reshares (default from config)
            
        Returns:
            dict with spread results including path, stats per step, etc.
        """
        if max_depth is None:
            max_depth = self.max_depth
        if spread_prob is None:
            spread_prob = self.spread_probability

        # Reset the network statuses
        self.network.reset_statuses()

        # BFS initialization
        queue = deque([(start_node, 0)])  # (node, depth)
        visited = set([start_node])
        spread_path = [start_node]
        spread_per_step = []  # Nodes infected at each BFS level
        current_step_nodes = []
        current_depth = 0
        infection_order = [(start_node, 0)]  # (node, time_step)

        # Mark start node as infected
        self.graph.nodes[start_node]["status"] = "infected"
        self.graph.nodes[start_node]["shared"] = True
        self.graph.nodes[start_node]["infection_time"] = 0

        while queue:
            node, depth = queue.popleft()

            if depth > max_depth:
                break

            # Track step transitions
            if depth > current_depth:
                spread_per_step.append(len(current_step_nodes))
                current_step_nodes = []
                current_depth = depth

            # Get neighbors and attempt to spread
            neighbors = list(self.graph.neighbors(node))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited and depth + 1 <= max_depth:
                    # Each node independently decides whether to reshare
                    role = self.graph.nodes[neighbor]["role"]

                    # Role-based spread probability adjustments
                    node_spread_prob = spread_prob
                    if role == "influencer":
                        node_spread_prob = min(1.0, spread_prob * 1.5)  # Influencers share more
                    elif role == "fact_checker":
                        node_spread_prob = spread_prob * 0.3  # Fact-checkers share less
                    elif role == "moderator":
                        node_spread_prob = spread_prob * 0.2  # Moderators are cautious

                    if random.random() < node_spread_prob:
                        visited.add(neighbor)
                        spread_path.append(neighbor)
                        current_step_nodes.append(neighbor)
                        infection_order.append((neighbor, depth + 1))

                        # Update node status
                        self.graph.nodes[neighbor]["status"] = "infected"
                        self.graph.nodes[neighbor]["shared"] = True
                        self.graph.nodes[neighbor]["infection_time"] = depth + 1

                        queue.append((neighbor, depth + 1))

                    # Track exposure even if not shared
                    self.graph.nodes[neighbor]["exposure_count"] += 1

        # Append last step
        if current_step_nodes:
            spread_per_step.append(len(current_step_nodes))

        # Calculate spread metrics
        total_reached = len(spread_path)
        penetration_rate = total_reached / self.network.num_nodes
        max_depth_reached = max(d for _, d in infection_order) if infection_order else 0

        # Calculate viral coefficient
        viral_coeff = 0
        if len(spread_path) > 1:
            total_secondary = sum(spread_per_step[1:]) if len(spread_per_step) > 1 else 0
            viral_coeff = total_secondary / max(1, spread_per_step[0]) if spread_per_step else 0

        return {
            "claim": claim_text,
            "start_node": start_node,
            "spread_path": spread_path,
            "spread_per_step": spread_per_step,
            "infection_order": infection_order,
            "total_reached": total_reached,
            "penetration_rate": round(penetration_rate * 100, 2),
            "max_depth_reached": max_depth_reached,
            "viral_coefficient": round(viral_coeff, 2),
            "total_exposures": sum(
                self.graph.nodes[n]["exposure_count"]
                for n in self.graph.nodes()
            ),
            "source_agent": "NeutralAgent",
        }

    def get_spread_summary(self, spread_result):
        """Format spread results into a human-readable summary."""
        sr = spread_result
        summary = f"""
📊 SPREAD SIMULATION RESULTS
{'='*50}
🚀 Starting Node: User_{sr['start_node']}
📡 Total Nodes Reached: {sr['total_reached']} / {self.network.num_nodes}
📈 Network Penetration: {sr['penetration_rate']}%
🔄 Max Spread Depth: {sr['max_depth_reached']} hops
📊 Viral Coefficient: {sr['viral_coefficient']}
👁️ Total Exposures: {sr['total_exposures']}
📋 Spread Path: {sr['spread_path'][:20]}{'...' if len(sr['spread_path']) > 20 else ''}
📶 Nodes per Step: {sr['spread_per_step']}
"""
        return summary
