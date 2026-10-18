# FinAI Agent IA d'Analyse Financière
FinAI est un agent conversationnel intelligent spécialisé dans l'analyse financière. 
Conçu pour surmonter les limites des LLM classiques, il combine un agent de raisonnement ReAct, un pipeline RAG multimodal (texte et vision) et un scoring multifactoriel adapté à différentes classes d'actifs.
Ce projet a été réalisé par : Chloé HAMON, Vadim ANNEQUIN, Félix DEPRAT, Solo DRAME et Mohamed-Ali BOUADLA.

Architecture Technique
Le système s'articule autour de composants open-source tournant entièrement en local via Ollama, garantissant rapidité, autonomie et absence de coûts d'API :   
LLM Local (Raisonnement Agent) : qwen2.5:7b (pilotage de la boucle ReAct).
LLM Local (Vision / RAG) : qwen2.5vl:3b (analyse visuelle des pages de rapports PDF, tableaux et graphiques).
Embeddings : all-MiniLM-L6-v2 (Hugging Face, 384 dimensions sur CPU).
Base Vectorielle : ChromaDB.
Interface Utilisateur : Streamlit.
Ordonnancement : scheduler.py pour la mise à jour continue des flux.

Structure du Projet
├── config.py                  # Centralisation de la configuration (actifs, chemins, modèles)

├── collecte.py                # Collecte des données (yFinance, flux RSS, ChromaDB)

├── agent.py                   # Agent conversationnel ReAct et mémoire fenêtrée

├── rag.py                     # Pipeline de recherche sémantique et traduction FR-EN

├── pipeline_pdf.py            # Ingestion des rapports PDF (texte + vision)

├── financial_tools_langchain.py # Registre des 14 outils LangChain `@tool`

├── scoring_global.py          # Modèle de scoring multifactoriel par classe d'actif

├── analyse_tech.py            # Indicateurs d'analyse technique (RSI, MACD, etc.)

├── analyse_fond.py            # Analyse fondamentale (ratios financiers yFinance)

├── macro.py                   # Indicateurs macroéconomiques (API FRED, VIX, courbe des taux)

├── scheduler.py               # Tâches de fond et mises à jour périodiques

└── app.py                     # Interface Web Streamlit

Modules Fonctionnels
Analyse Technique (analyse_tech.py) : Implémente 6 indicateurs chartistes (RSI, MACD, EWM, etc.).   
Analyse Fondamentale (analyse_fond.py) : Récupération de métriques via yFinance (P/E, P/B, dettes, marges)
Analyse Macroéconomique (macro.py) : Connexion à l'API FRED (18 séries US), suivi du VIX, du Fear & Greed Index (CNN) et de la courbe des taux 2Y-10Y.
Scoring Global (scoring_global.py) : Calcul d'un score pondéré selon le type d'actif :
Actions : Technique (35%), Fondamental (35%), Sentiment (15%), Macro (15%).
ETF : Technique (40%), Fondamental (20%), Sentiment (15%), Macro (25%).
Cryptomonnaies : Technique (45%), Fondamental (0%), Sentiment (35%), Macro (20%).
Matières premières : Technique (35%), Fondamental (0%), Sentiment (25%), Macro (40%).   

RAG Multimodal & Traduction : Indexation de rapports financiers PDF. Intégration d'un dictionnaire de traduction automatique FR/EN des termes financiers (ex: "chiffre d'affaires" $\rightarrow$ "total revenue") pour optimiser la recherche sémantique

Registre des 14 Outils LangChain
L'agent orchestre un ensemble de 14 outils spécialisés : 
recherche_financiere : Recherche dans les rapports annuels (RAG + ChromaDB).
get_cours_action / get_historique_action : Suivi des prix et performances yFinance.
get_donnees_financieres / get_pe_ratio : Extraction des métriques fondamentales.
analyser_action : Calculs d'indicateurs techniques.
score_global_action : Évaluation globale et recommandation de type BUY / SELL / NEUTRAL.
get_top_performers / get_correlation / get_news_action : Analyse comparative et actualités (RSS).   
calcul_marge_nette, calcul_croissance, calcul_ratio_endettement, verifier_alerte : Fonctions de calcul local.

Installation & Utilisation
Prérequis
Python 3.10 + Ollama installé localement avec les modèles requis :Bashollama pull qwen2.5:7b
ollama pull qwen2.5vl:3b

1. Installation des dépendancesCloner le dépôt et installer les bibliothèques nécessaires :Bashgit clone https://github.com/chloe-hamon/FinAI.git
cd FinAI
pip install -r requirements.txt

2. Configuration
   Vérifiez ou adaptez les paramètres dans le fichier config.py selon vos besoins (chemins de stockage, actifs surveillés, flux RSS)
   Lancer l'interface StreamlitDémarrez l'application web interactive :Bashstreamlit run app.py

Évaluation et Performances
Évaluation RAG : Le système intègre un module d'évaluation automatique (evaluation_rag_auto.py) noté sur 20 points s'appuyant sur 5 critères (pertinence, cohérence, exactitude, utilisation du RAG et clarté).
Routage Hybride : Pour réduire la latence, les requêtes fréquentes (comme les demandes de scores ou d'analyses techniques directes) contournent la boucle ReAct lourde pour appeler directement les outils ciblés.
