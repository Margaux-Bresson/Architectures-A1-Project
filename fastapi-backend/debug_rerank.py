"""Diagnostic du reranking : python debug_rerank.py "assurance" "Lave linge séchant"

Teste les 3 stratégies (voir rerank.STRATEGIES) et affiche, pour chacune, la réponse du
modèle, les tokens dont Ollama renvoie les logprobs, et p_yes / p_no.
"""
import math
import sys

import rerank

query = sys.argv[1] if len(sys.argv) > 1 else "assurance"
document = sys.argv[2] if len(sys.argv) > 2 else "Lave linge séchant"

print(f"Modèle : {rerank.RERANK_MODEL} | requête : {query!r} | document : {document!r}\n")

for strategy in rerank.STRATEGIES:
    try:
        body = rerank.ask_ollama(query, document, strategy)
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"[{strategy}] ERREUR : {exc}\n")
        continue

    logprobs = body["logprobs"]
    print(f"[{strategy}] réponse={body['content']!r} | tokens avec logprobs={len(logprobs)}")
    print("   tokens :", [e["token"] for e in logprobs])
    if logprobs:
        first = logprobs[0].get("top_logprobs") or []
        print("   alternatives du 1er token :", [(a["token"], round(math.exp(a["logprob"]), 4)) for a in first[:8]])

    probabilities = rerank.yes_no_from_logprobs(logprobs, body["content"], strategy)
    if probabilities:
        p_yes, p_no = probabilities
        print(f"   => p_yes={p_yes:.6f} p_no={p_no:.6f} score={p_yes / (p_yes + p_no):.6f}  OK")
    else:
        print("   => token de la réponse sans logprobs yes/no : stratégie inutilisable")
    print()
