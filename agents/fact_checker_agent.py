import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, FACT_CHECKER_SYSTEM_PROMPT, TEMPERATURE

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


class FactCheckerAgent:
    def __init__(self):
        self.llm = None
        if GROQ_API_KEY:
            try:
                self.llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=GROQ_MODEL,
                    temperature=0.3,
                    max_tokens=512,
                )
            except Exception:
                self.llm = None

    def check_graph(self, network) -> dict:
     
        G = network.graph
        fc_nodes = network.get_fact_checker_nodes()

        claim_text = ""
        for n in G.nodes():
            if G.nodes[n].get("claim_text"):
                claim_text = G.nodes[n]["claim_text"]
                break

        if not claim_text:
            return {
                "success": True,
                "verdict": "Unverified",
                "confidence": 0.0,
                "nodes_checked": 0,
                "nodes_warned": 0,
                "evidence": "No claim found in the network",
                "red_flags": [],
                "active_checkers": 0,
                "search_provider": "duckduckgo",
                "search_summary": "No claim present, so no web search executed.",
                "search_sources": [],
            }

     
        verdict_result = self.verify_claim(claim_text)

        nodes_checked = 0
        nodes_warned = 0
        warned_node_ids = []
        active_checkers = 0

        for fc_node in fc_nodes:
            if G.nodes[fc_node]["exposure_count"] > 0 or any(
                G.nodes[nb]["status"] in ("infected", "influenced")
                for nb in G.neighbors(fc_node)
            ):
                active_checkers += 1
                G.nodes[fc_node]["status"] = "warned"
                G.nodes[fc_node]["warning_label"] = True

               
                for nb in G.neighbors(fc_node):
                    if G.nodes[nb]["status"] in ("infected", "influenced"):
                        nodes_checked += 1
                        if verdict_result["verdict"] in ("Fake", "Unverified"):
                            G.nodes[nb]["warning_label"] = True
                            G.nodes[nb]["status"] = "warned"
                            nodes_warned += 1
                            warned_node_ids.append(nb)

        return {
            "success": True,
            "claim": claim_text,
            "verdict": verdict_result["verdict"],
            "confidence": verdict_result["confidence"],
            "evidence": verdict_result["evidence"],
            "red_flags": verdict_result.get("red_flags", []),
            "search_provider": verdict_result.get("search_provider", "duckduckgo"),
            "search_summary": verdict_result.get("search_summary", ""),
            "search_sources": verdict_result.get("search_sources", []),
            "nodes_checked": nodes_checked,
            "nodes_warned": nodes_warned,
            "warned_node_ids": warned_node_ids,
            "active_checkers": active_checkers,
            "total_fact_checkers": len(fc_nodes),
        }

    def verify_claim(self, claim_text: str) -> dict:
        search_evidence, search_sources, search_provider = self._live_web_search(claim_text)

        if self.llm:
            try:
                prompt = (
                    f'Analyze the following claim for truthfulness:\n\n'
                    f'CLAIM: "{claim_text}"\n\n'
                    f'WEB SEARCH EVIDENCE (provider={search_provider}):\n{search_evidence}\n\n'
                    f'Respond ONLY with valid JSON in the specified format.'
                )
                messages = [
                    SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                result = self._parse_verdict(response.content)
                result["claim"] = claim_text
                result["search_evidence"] = search_evidence
                result["search_provider"] = search_provider
                result["search_summary"] = f"Searched via {search_provider} and found {len(search_sources)} relevant sources."
                result["search_sources"] = search_sources
                return result
            except Exception:
                pass

        return self._fallback_verdict(claim_text, search_evidence, search_sources, search_provider)

    @staticmethod
    def _format_search_evidence(claim_text: str, sources: list[dict[str, str]], provider: str) -> str:
        lines = [
            f"Search provider: {provider}",
            f"Query: {claim_text}",
        ]
        if not sources:
            lines.append("No reliable search results were returned.")
            return "\n".join(lines)

        for idx, src in enumerate(sources, start=1):
            lines.append(
                f"[{idx}] {src.get('title', 'Untitled')} | {src.get('url', 'N/A')} | {src.get('snippet', '')}"
            )
        return "\n".join(lines)

    def _live_web_search(self, claim_text: str, max_results: int = 6) -> tuple[str, list[dict[str, str]], str]:
        provider = "duckduckgo"
        if DDGS is None:
            summary = (
                "DuckDuckGo search package is not available. "
                "Install duckduckgo-search to enable live web lookup."
            )
            return summary, [], provider

        queries = [
            claim_text,
            f"{claim_text} fact check",
            f"India election fact check {claim_text[:100]}",
        ]

        sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        try:
            with DDGS() as ddgs:
                for query in queries:
                    for item in ddgs.text(query, max_results=max_results):
                        url = (item.get("href") or "").strip()
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        sources.append({
                            "title": (item.get("title") or "Untitled").strip(),
                            "url": url,
                            "snippet": (item.get("body") or "").strip(),
                            "query": query,
                        })
                        if len(sources) >= max_results:
                            break
                    if len(sources) >= max_results:
                        break
        except Exception as e:
            return f"DuckDuckGo search failed: {e}", [], provider

        summary = self._format_search_evidence(claim_text, sources, provider)
        return summary, sources, provider

    @staticmethod
    def _parse_verdict(response_text: str) -> dict:
        try:
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
            verdict = result.get("verdict", "Unverified")
            if verdict not in ("Real", "Fake", "Unverified"):
                verdict = "Unverified"
            return {
                "verdict": verdict,
                "confidence": float(result.get("confidence", 0.5)),
                "evidence": result.get("evidence", "No evidence provided"),
                "red_flags": result.get("red_flags", []),
            }
        except (json.JSONDecodeError, ValueError):
            lower = response_text.lower()
            if "fake" in lower:
                v = "Fake"
            elif "real" in lower or "true" in lower:
                v = "Real"
            else:
                v = "Unverified"
            return {
                "verdict": v,
                "confidence": 0.5,
                "evidence": response_text[:200],
                "red_flags": ["Could not parse structured response"],
            }

    @staticmethod
    def _fallback_verdict(
        claim_text: str,
        evidence: str,
        sources: list[dict[str, str]],
        search_provider: str,
    ) -> dict:
        red_flags = []
        score = 0.5
        lower = claim_text.lower()
        triggers = ["breaking", "secret", "leaked", "shocking", "banned",
                     "they don't want you to know", "100%", "miracle"]
        for t in triggers:
            if t in lower:
                red_flags.append(f'Contains trigger word: "{t}"')
                score += 0.08
        verdict = "Fake" if score >= 0.6 else "Unverified"
        return {
            "claim": claim_text,
            "verdict": verdict,
            "confidence": round(min(score, 0.95), 2),
            "evidence": "Rule-based analysis (LLM unavailable). " + evidence,
            "red_flags": red_flags or ["No specific red flags detected"],
            "search_evidence": evidence,
            "search_provider": search_provider,
            "search_summary": f"Searched via {search_provider} and found {len(sources)} relevant sources.",
            "search_sources": sources,
        }

    @staticmethod
    def get_verdict_summary(result: dict) -> str:
        emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(result["verdict"], "❓")
        flags = "\n".join(f"  🚩 {f}" for f in result.get("red_flags", []))
        return (
            f"\n🔍 FACT-CHECK RESULTS\n{'='*50}\n"
            f"{emoji} Verdict : {result['verdict']}\n"
            f"📊 Confidence: {result['confidence']*100:.0f}%\n"
            f"📝 Evidence  : {result['evidence']}\n"
            f"🚩 Red Flags :\n{flags or '  None'}\n"
            f"📋 Nodes Checked: {result.get('nodes_checked', 0)}\n"
            f"⚠️ Nodes Warned : {result.get('nodes_warned', 0)}\n"
            f"🔍 Active Checkers: {result.get('active_checkers', 0)} / "
            f"{result.get('total_fact_checkers', 0)}\n"
        )
