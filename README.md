# DecisionLLM

PoC di decision model "System 1" costruito su un LLM locale qualsiasi: riceve uno stato (un testo) e delle
domande tipizzate e restituisce risposte tipizzate con probabilità, con un solo forward pass per domanda e
senza generare testo.

Il meccanismo: le opzioni vengono elencate come `A`, `B`, `C`... e si legge la distribuzione del token
successivo ristretta a quelle lettere. La softmax sulle lettere dà la probabilità di ogni classe, l'argmax è
la decisione. Lo schema segue *Robots That Ask For Help* (KnowNo, Ren et al. 2023,
[arXiv:2307.01928](https://arxiv.org/abs/2307.01928)): prompt a scelta multipla, opzione "nessuna delle altre",
permutazione delle opzioni e conformal prediction per decidere quando il modello è sicuro.

Demo online: **[davidemodolo.github.io/DecisionLLM](https://davidemodolo.github.io/DecisionLLM/)**, con risposte
registrate dai modelli veri (vedi [Demo statica](#demo-statica)).

## Contenuto

| Percorso | Cosa fa |
|---|---|
| `demo.ipynb` | Notebook passo passo: logits, vincolo sulle lettere, batch, bias posizionale, scramble, conformal prediction, calibrazione, decisioni tipizzate |
| `core.py` | Motore: modelli, prompt, batching, scramble, aggregazioni, metriche, calibrazione conforme |
| `server.py` | API FastAPI e server della UI |
| `systems.py` | I decision model (Clef-flash, Laya, Jev, Gemini Decision 1) con la stessa interfaccia del motore, per server e UI |
| `static/index.html` | UI web |
| `download_models.py` | Scarica i pesi dei modelli e il benchmark nella cache di Hugging Face |
| `make_test_questions.py` | Genera `data/test_questions.json` (256 domande italiane con risposta nota) |
| `bench/run_bench.py` | Benchmark dei modelli su Global-MMLU-Lite italiano e stress test |
| `bench/compare.py` | Confronto con Laya, un classificatore NLI zero-shot, GLiClass, GLiDE, vela3.1-fast, d3-lite, Clef-flash, Jev e Gemini Decision 1 (API) su dataset italiani di classificazione |
| `bench/report.py` | Grafici (`bench/plots/`) e tabelle (`bench/RESULTS.md`) dai risultati |
| `pages/` | Demo statica per GitHub Pages: registrazione delle risposte, API simulata nel browser, build del sito |
| `tests/` | Test della logica pura, dell'API e di equivalenza dei percorsi di forward su modelli minuscoli generati dalle config reali |

## Avvio con Docker

Serve una GPU NVIDIA con [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose build
docker compose run --rm decisionllm python download_models.py   # tutti i modelli, circa 25 GB
docker compose run --rm decisionllm python download_models.py qwen3.5-2b lfm2.5-1.2b   # solo alcuni
docker compose up -d                                              # http://localhost:8000
```

I pesi e la cache dei kernel Triton finiscono nel volume `model-cache`. Un modello non ancora scaricato viene
scaricato al primo caricamento. `HF_TOKEN` nell'ambiente e le chiavi API del file `.env` (`CLASSIFIER_DEV_API`
per Jev, `GOOGLE_API_KEY` per Gemini) vengono passati al container. L'immagine include Laya e GLiClass. `data/`,
`bench/data/` e `bench/results/` sono montati dalla cartella del progetto, quindi calibrazioni, benchmark e
risultati restano sull'host.

Il benchmark nel container: `docker compose run --rm decisionllm python bench/run_bench.py eval`.

## Avvio locale

```bash
uv sync
uv run python server.py            # http://127.0.0.1:8000
uv run jupyter lab demo.ipynb
uv run pytest
```

`server.py --preload qwen3.5-2b` carica un modello all'avvio. Altrimenti viene caricato alla prima esecuzione
o con il pulsante "Carica". Un solo modello resta in VRAM alla volta. Laya richiede `uv sync --group baselines`;
Jev e Gemini richiedono la loro chiave in `.env`.

## Modelli

Pensati per una GPU da 8 GB. I parametri sono quelli del repository (per Qwen3.5 includono l'encoder visivo,
che non viene caricato).

| Chiave | Repo | Parametri | Precisione |
|---|---|---|---|
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | 0.87B | bf16 |
| `lfm2.5-1.2b` | `LiquidAI/LFM2.5-1.2B-Instruct` | 1.17B | bf16 |
| `qwen3.5-2b` | `Qwen/Qwen3.5-2B` | 2.27B | bf16 |
| `granite-4.2-3b` | `ibm-granite/granite-4.2-3b` | 3.66B | 4-bit NF4 |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | 4.66B | 4-bit NF4 |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | 9.65B | 4-bit NF4, riempie la GPU |

La lista è in `core.MODELS`. Per Qwen3.5 i kernel di `flash-linear-attention` (Triton) accelerano i layer
gated-delta: il primo forward di un nuovo processo compila i kernel (circa 40 s), poi la cache su disco li riusa.
Le lunghezze dei prompt sono arrotondate a multipli di 32 token, così i kernel vengono compilati per poche forme.

Qwen3.5 9B tiene la tabella degli embedding (vocabolario da 248k token, 1.9 GB in bf16) in RAM: il lookup avviene
su CPU e solo i vettori risultanti passano alla GPU. Così il modello entra in 8 GB con margine per i batch.

## Come funziona

**Tipi di domanda.**

- `choice`: scelta tra opzioni. Per default si aggiunge "Nessuna delle altre" come ultima opzione.
- `score`: livelli ordinati dal più basso al più alto (2-10). Il punteggio è la media dei livelli 1..n
  pesata per le probabilità; `score_norm` lo riporta in [0, 1].
- `yesno`: opzioni Sì/No; il risultato è `p_yes`.

**Stato.** Un testo di contesto condiviso, messo in testa al prompt di ogni domanda (`Contesto: ...`). Ogni domanda
è un prompt separato e tutte vengono valutate negli stessi batch.

**Prompt.** Chat template del modello, thinking disattivato, istruzione *"Quale opzione è corretta? Rispondi con
una sola lettera (A, B, C, D o E), senza altro testo né formattazione."* e risposta che inizia con `Risposta:`.
È lo stile che tiene la massa di probabilità sulle lettere sopra il 98% sui modelli fino a 4B e al 95% sul 9B (vedi
[`bench/RESULTS.md`](bench/RESULTS.md)). Due dettagli contano:

- il prompt termina senza spazio: dopo `"Risposta: "` il token `" B"` non può più uscire e la massa sulle lettere scende, di quanto dipende dal modello (il notebook lo mostra);
- l'elenco esplicito delle lettere e "senza formattazione" evitano che Qwen3.5 inizi con `**` (grassetto markdown).

**Lettere.** Per ogni lettera si considerano i token `"B"` e `" B"` e se ne sommano le probabilità. La **massa
sulle lettere** è la probabilità totale delle lettere sul vocabolario completo: se è bassa, il modello voleva
rispondere altro e la distribuzione rinormalizzata è poco affidabile.

**Batch.** I prompt sono ordinati per lunghezza e raggruppati con un budget di token (`batch_tokens`, default
2048, cioè righe × lunghezza del più lungo) e un massimo di righe (`batch_size`, default 64). Il prefill è
limitato dal calcolo: batch di prompt di lunghezza simile sprecano poco padding e oltre circa 2048 token per
batch il throughput scende (vedi [`bench/RESULTS.md`](bench/RESULTS.md)). Padding a
sinistra e testa del vocabolario calcolata solo sull'ultima posizione; del vocabolario si conservano solo le
log-probabilità delle lettere e l'argmax. In caso di OOM il batch si divide a metà e si riprova. Massimo 256
domande per richiesta.

**Cache del prefisso.** Le permutazioni di una domanda condividono tutto il testo prima delle opzioni, e con uno
stato tutte le domande condividono il contesto. Il motore calcola una volta la cache del prefisso comune (KV per i
layer di attention, stato ricorrente per i layer lineari o convolutivi) e la riusa per tutte le continuazioni:

- con uno stato condiviso la cache del contesto è unica per tutta la richiesta;
- altrimenti ogni domanda ha la sua cache, riusata dalle sue K permutazioni.

Le probabilità coincidono con il forward completo (verificato in fp32 dai test; in bf16 l'argmax coincide nel
98% dei casi, il resto è rumore numerico). Il guadagno cresce con la parte comune: da 2.5× a 3.3× per domanda con uno
stato lungo, circa +20% di throughput su Global-MMLU (39% di token in comune). Con domande brevi e poco testo in
comune copiare la cache costa più del prefill risparmiato, quindi la cache si attiva solo quando risparmia almeno
il 25% dei token (`core.MIN_PREFIX_SAVING`). Si controlla con `prefix_cache`, attivo per default nell'API e nella UI.

**Set-based attention** (solo Granite, che ha solo layer di attention). Segue *Set-Based Prompting* (McIlroy-Young et al.,
2024): le opzioni partono tutte dalla stessa posizione e non si vedono tra loro, quindi il prompt è invariante all'ordine delle opzioni con
un solo forward. Si ottiene con una maschera di attention 4D e `position_ids` espliciti. Il modello non è
addestrato con questa maschera e l'accuratezza scende (vedi [`bench/RESULTS.md`](bench/RESULTS.md)); resta come
opzione dimostrativa da confrontare con lo scramble. I modelli con layer ricorrenti la rifiutano, perché uno
stato ricorrente dipende dall'ordine dei token.

**Scramble answers.** Ogni domanda viene posta K volte permutando tutte le opzioni, compresa "Nessuna delle
altre". Le prime permutazioni sono rotazioni cicliche: con K=5 e 5 opzioni ogni risposta occupa ogni lettera
esattamente una volta e il bias posizionale si cancella. Le probabilità vengono riportate all'ordine originale e
aggregate:

- `geomean`: media geometrica rinormalizzata (prodotto di esperti), penalizza un'opzione che anche una sola variante ritiene molto improbabile;
- `mean`: media aritmetica;
- `weighted`: media pesata con la massa sulle lettere di ogni variante;
- `vote`: voto di maggioranza sugli argmax, con la media come spareggio.

La **consistenza** è la frazione di varianti il cui argmax coincide con la risposta finale.

**Calibrazione.** Sulle domande con risposta nota si calcolano ECE (scarto medio tra confidenza e accuratezza per
fasce di confidenza), Brier (errore quadratico rispetto al one-hot) e NLL.

**Conformal prediction.** Su un set di calibrazione con risposte note si calcola `κ = 1 - p(risposta
corretta)` e `q̂`, il quantile di livello `⌈(N+1)(1-ε)⌉/N`. Su una nuova domanda il prediction set contiene le
opzioni con `p ≥ 1 - q̂`: la risposta giusta vi appartiene con probabilità ≥ `1-ε`. Se il set ha una sola
opzione il modello decide, altrimenti chiede aiuto. Le calibrazioni sono salvate in `data/calibrations.json`,
una per combinazione di modello, prompt, scramble e aggregazione.

## UI

Material 3, tema chiaro e scuro, quattro sezioni. Il selettore in alto raggruppa i nostri LLM, i decision model
sulla GPU (Clef-flash, Laya) e quelli via API (Jev, Gemini Decision 1), con accuratezza media e millisecondi per
domanda del benchmark. Un decision model a cui manca la chiave API o il pacchetto resta visibile, disattivato,
con il motivo. I decision model rispondono con una passata nel loro formato di domande tipizzate, quindi scramble,
stile del prompt e opzioni di velocità valgono solo per i nostri LLM.

- **Decidi**: contesto, domande ed esecuzione. Ogni domanda ha un tipo (scelta, punteggio, sì / no), le opzioni o i
  livelli, "Nessuna delle altre" facoltativa e una risposta attesa facoltativa. La risposta compare sotto la
  domanda: opzione scelta, probabilità di ogni opzione, sicurezza (o "decide da solo" / "chiede aiuto" se il
  modello è calibrato), coerenza tra le permutazioni e il dettaglio di ogni permutazione con la lettera mostrata
  e il token che il modello avrebbe scritto senza vincoli. Quattro esempi pronti (segnalazione al Comune, email
  all'ufficio tributi, recensione, quiz), import ed export JSON, Ctrl+Invio per eseguire.
- **Valuta**: accuratezza, ECE, Brier e velocità sulle domande di test o sulle proprie domande con risposta
  attesa, con diagramma di affidabilità, istogramma della confidenza, preferenza per le lettere, accuratezza per
  categoria e tabella delle risposte. "Con e senza scramble" esegue le due configurazioni affiancate. Qui si
  calibra la conformal prediction. Lo storico delle esecuzioni resta nel browser.
- **Modelli**: i risultati di `bench/results` (accuratezza e calibrazione rispetto ai parametri, conformal,
  set-attention, velocità a caldo e cache del prefisso, tabella completa).
- **Come funziona**: una domanda passo per passo, dal prompt al token, alle probabilità sulle lettere, alle
  permutazioni, alla decisione.

Le impostazioni (scramble e aggregazione, cache del prefisso, set-attention, stile del prompt, budget dei batch)
sono nel pannello laterale aperto dall'icona in alto a destra.

## API

| Metodo | Percorso | Corpo |
|---|---|---|
| GET | `/api/config` | modelli, stili di prompt, GPU, calibrazioni salvate |
| POST | `/api/load` | `{"model"}` |
| POST | `/api/score` | `{"model", "state"?, "questions": [{"type", "question", "options", "none_option"?, "answer"?, "state"?}], "scramble", "aggregation", "batch_size", "batch_tokens", "prompt_style", "prefix_cache", "set_attention"}` |
| POST | `/api/calibrate` | come `/api/score` più `"epsilon"` e `"source"`: `"questions"` o `"global-mmlu-dev"` |
| POST | `/api/decide` | decisioni tipizzate, vedi sotto |
| GET | `/api/test-questions` | le 256 domande di test |
| GET | `/api/bench` | i risultati del benchmark in `bench/results`, per fase e modello |

`/api/decide` prende uno stato e una mappa di domande e restituisce una risposta tipizzata per chiave:

```bash
curl -s localhost:8000/api/decide -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-4b",
  "scramble": 5,
  "state": "Il risotto era ottimo, ma abbiamo aspettato un ora e il conto era alto.",
  "questions": {
    "critica": {"type": "choice", "instructions": "Qual è l aspetto più criticato?",
                "criteria": {"cibo": "Qualità del cibo", "attesa": "Tempi di attesa", "prezzo": "Prezzo"}},
    "soddisfazione": {"type": "score", "instructions": "Quanto è soddisfatto il cliente?",
                      "criteria": ["Per niente", "Poco", "Abbastanza", "Molto"]},
    "rimborso": {"type": "yesno", "instructions": "Il cliente chiede un rimborso?"}
  }
}'
```

Risposta reale (Qwen3.5 4B in 4 bit, a caldo, probabilità arrotondate, calibrazione su Global-MMLU dev con ε = 0.1):

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "answers": {
    "critica": {"probabilities": {"cibo": 0.02, "attesa": 0.86, "prezzo": 0.13}, "confidence": 0.86, "choice": "attesa", "prediction_set": ["attesa", "prezzo"], "decided": false},
    "soddisfazione": {"probabilities": {"Per niente": 0.21, "Poco": 0.66, "Abbastanza": 0.08, "Molto": 0.05}, "confidence": 0.66, "score": 1.96, "score_norm": 0.32, "level": "Poco", "prediction_set": ["Per niente", "Poco"], "decided": false},
    "rimborso": {"probabilities": {"yes": 0.67, "no": 0.33}, "confidence": 0.67, "p_yes": 0.67, "prediction_set": ["yes", "no"], "decided": false}
  },
  "timing": {"total_ms": 706, "forward_ms": 700, "prompts": 12, "tokens": 783}
}
```

Il modello individua l'attesa come critica principale e una soddisfazione bassa; sul rimborso sbaglia, con una
confidenza del 67%. `prediction_set` e `decided` compaiono quando esiste una calibrazione per quel modello e
quella configurazione (`/api/calibrate`): un'opzione entra nel set se supera la soglia calibrata, qui l'8.2%.
Con questa soglia, prudente perché calibrata su domande di cultura generale, il modello chiede aiuto su tutte e
tre le domande. Una calibrazione su domande del proprio dominio dà una soglia adatta a quel compito.

`allow_none: true` su una domanda `choice` aggiunge l'opzione `none`. `scramble`, `aggregation`, `prefix_cache` e `set_attention` funzionano come in `/api/score`.

## Benchmark

```bash
uv run python bench/run_bench.py prompt     # confronto degli stili di prompt (dev)
uv run python bench/run_bench.py eval       # calibrazione su dev, valutazione su test
uv run python bench/run_bench.py stress     # tempi a caldo: budget di token, cache del prefisso, stato lungo
uv run python bench/report.py               # grafici e bench/RESULTS.md
```

Dataset: [Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite), split italiano
(215 dev, 400 test, 4 opzioni). Al 20% delle domande viene tolta l'opzione corretta, così la risposta giusta
diventa "Nessuna delle altre". Risultati, grafici di accuratezza, calibrazione e velocità rispetto al numero di
parametri sono in [`bench/RESULTS.md`](bench/RESULTS.md).

### Confronto con i decision model

```bash
uv sync --group baselines
uv run python bench/compare.py baselines    # Laya, NLI, GLiClass, GLiDE, vela, d3-lite, Clef-flash, Jev e Gemini (chiavi in .env)
uv run python bench/compare.py llm          # i nostri modelli, 1 passata e 5 permutazioni
uv run python bench/report.py
```

`bench/compare.py` pone le stesse domande a sette modelli open source che restituiscono probabilità con una sola
passata, a due decision model via API (Jev e Gemini Decision 1) e ai nostri LLM. Le chiavi vanno in `.env`
(`CLASSIFIER_DEV_API` per Jev, `GOOGLE_API_KEY` per Gemini). Jev riceve `share_data: false`. Le risposte restano
in cache in `bench/data/jev_cache.json` e `bench/data/gemini_cache.json`, così un nuovo run non ripete le
chiamate. Le richieste a Gemini restano sotto i limiti del livello gratuito (180 richieste e 45k token al minuto):

| Sistema | Repo | Parametri | Come risponde |
|---|---|---|---|
| Laya multilingual | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (`multilingual`) | 322M | decision model su mmBERT: stato e domande tipizzate (scelta, punteggio, sì / no), come Jev |
| bge-m3 zeroshot v2.0 | [`MoritzLaurer/bge-m3-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0) | 568M | cross-encoder NLI: una passata per opzione, probabilità di implicazione normalizzate |
| GLiClass x-base | [`knowledgator/gliclass-x-base`](https://huggingface.co/knowledgator/gliclass-x-base) | 300M | mDeBERTa: testo e tutte le opzioni in una passata, un punteggio per opzione |
| GLiDE 0.4B | [`fastino/GLiDE-0.4B`](https://huggingface.co/fastino/GLiDE-0.4B) | 0.4B | decision model di Fastino su encoder ModernBERT (Ettin): stato e domande tipizzate, un punteggio per opzione; licenza Fastino Community (uso commerciale in produzione libero sotto 10 M$ di fatturato annuo, con la dicitura "Built with Fastino GLiDE") |
| vela3.1-fast | [`vllm-sr/vela3.1-fast`](https://huggingface.co/vllm-sr/vela3.1-fast) | 380M | decision model di vLLM Semantic Router su encoder UltraBERT, addestrato in 15 lingue; scelta e sì / no (il punteggio passa come scelta tra i livelli) |
| d3-lite | [`vllm-sr/d3-lite`](https://huggingface.co/vllm-sr/d3-lite) | 0.85B | decision model di vLLM Semantic Router su Qwen3.5 0.8B, stato e domande tipizzate come Jev |
| Clef-flash (4-bit) | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) | 9.65B | decision model di Cloudflare: Qwen3.5 9B con una testa che dà un logit per opzione, stato e domande tipizzate come Jev; layer in NF4 sulla nostra GPU |
| Jev (API) | [classifier.dev](https://classifier.dev) `/v1/systemone`, `jev-latest` | – | il decision model di TypeSafe, stato e domande tipizzate; tempi con la rete |
| Gemini Decision 1 (API) | [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api) `/v1beta/decisions`, `gemini-decision-1` | – | il decision model di Google, stato e domande tipizzate come Jev; tempi con la rete |

I dataset sono in italiano e coprono i tre tipi di domanda:

- [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) it-IT: richieste a un assistente vocale, scelta tra 5 intent (fino a 2 dello stesso dominio);
- [SIB-200](https://huggingface.co/datasets/Davlan/sib200) ita_Latn: argomento di una frase tra 7, e la stessa frase come domanda sì / no;
- [tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) italiano: punteggio su 3 livelli;
- le 256 domande locali e Global-MMLU-Lite, come riferimento sulle domande di conoscenza.

Ogni sistema produce una distribuzione sulle opzioni della domanda, valutata con le stesse metriche. Sullo split dev
si fittano una temperatura (ECE con T) e la soglia della conformal prediction. I tempi sono a caldo, sulla stessa
GPU. Il confronto è nella pagina Modelli della UI e in [`bench/RESULTS.md`](bench/RESULTS.md).

## Demo statica

La demo su GitHub Pages è la stessa UI, con `pages/demo.js` al posto del server. `pages/record.py` fa rispondere
tutti i modelli, con 10 permutazioni per domanda, agli esempi, alle 256 domande di test e alla domanda di "Come
funziona", e salva in `pages/data/` le probabilità di ogni permutazione e i tempi misurati a caldo. Nel browser la
demo ricostruisce da questi dati qualsiasi numero di permutazioni da 1 a 10 (le permutazioni con K minore sono le
prime di quelle con K = 10) e ricalcola aggregazioni, metriche e calibrazione conforme come `core.py`. Le domande
nuove ricevono probabilità casuali e sono segnate come simulate. La pagina Modelli legge `bench/results`.

```bash
uv run python pages/record.py --models qwen3.5-0.8b   # risposte registrate (GPU), default tutti
python3 pages/build.py             # sito in _site/
python3 -m http.server -d _site    # anteprima su http://localhost:8000
```

A ogni push su `main` il workflow `.github/workflows/pages.yml` costruisce il sito e lo pubblica.
