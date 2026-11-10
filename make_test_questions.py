"""Generates data/test_questions.json: 256 Italian multiple-choice questions with known answers."""
import json
import random
from pathlib import Path

N_QUESTIONS = 256
NONE_RATE = 0.2

CAPITALS = {
    "Francia": "Parigi", "Germania": "Berlino", "Spagna": "Madrid", "Portogallo": "Lisbona",
    "Austria": "Vienna", "Svizzera": "Berna", "Belgio": "Bruxelles", "Paesi Bassi": "Amsterdam",
    "Polonia": "Varsavia", "Grecia": "Atene", "Svezia": "Stoccolma", "Norvegia": "Oslo",
    "Finlandia": "Helsinki", "Danimarca": "Copenaghen", "Irlanda": "Dublino", "Ungheria": "Budapest",
    "Romania": "Bucarest", "Bulgaria": "Sofia", "Croazia": "Zagabria", "Slovenia": "Lubiana",
    "Repubblica Ceca": "Praga", "Slovacchia": "Bratislava", "Russia": "Mosca", "Ucraina": "Kiev",
    "Turchia": "Ankara", "Egitto": "Il Cairo", "Marocco": "Rabat", "Kenya": "Nairobi",
    "Giappone": "Tokyo", "Cina": "Pechino", "Corea del Sud": "Seul", "India": "Nuova Delhi",
    "Thailandia": "Bangkok", "Australia": "Canberra", "Canada": "Ottawa", "Stati Uniti": "Washington",
    "Messico": "Città del Messico", "Brasile": "Brasilia", "Argentina": "Buenos Aires", "Perù": "Lima",
}
REGIONS = {
    "Piemonte": "Torino", "Valle d'Aosta": "Aosta", "Lombardia": "Milano", "Trentino-Alto Adige": "Trento",
    "Veneto": "Venezia", "Friuli-Venezia Giulia": "Trieste", "Liguria": "Genova", "Emilia-Romagna": "Bologna",
    "Toscana": "Firenze", "Umbria": "Perugia", "Marche": "Ancona", "Lazio": "Roma",
    "Abruzzo": "L'Aquila", "Molise": "Campobasso", "Campania": "Napoli", "Puglia": "Bari",
    "Basilicata": "Potenza", "Calabria": "Catanzaro", "Sicilia": "Palermo", "Sardegna": "Cagliari",
}
ELEMENTS = {
    "Idrogeno": "H", "Elio": "He", "Carbonio": "C", "Azoto": "N", "Ossigeno": "O", "Sodio": "Na",
    "Magnesio": "Mg", "Alluminio": "Al", "Silicio": "Si", "Fosforo": "P", "Zolfo": "S", "Cloro": "Cl",
    "Potassio": "K", "Calcio": "Ca", "Ferro": "Fe", "Rame": "Cu", "Zinco": "Zn", "Argento": "Ag",
    "Oro": "Au", "Mercurio": "Hg", "Piombo": "Pb", "Stagno": "Sn", "Uranio": "U", "Neon": "Ne",
    "Iodio": "I", "Platino": "Pt", "Nichel": "Ni", "Cobalto": "Co", "Litio": "Li", "Fluoro": "F",
}
OPPOSITES = {
    "caldo": "freddo", "alto": "basso", "veloce": "lento", "grande": "piccolo", "pieno": "vuoto",
    "aperto": "chiuso", "facile": "difficile", "giovane": "anziano", "ricco": "povero", "forte": "debole",
    "chiaro": "scuro", "lungo": "corto", "pesante": "leggero", "vicino": "lontano", "dolce": "amaro",
    "nuovo": "vecchio", "felice": "triste", "presto": "tardi", "dentro": "fuori", "sopra": "sotto",
    "vincere": "perdere", "comprare": "vendere", "entrare": "uscire", "accendere": "spegnere",
    "ricordare": "dimenticare",
}
WORKS = {
    "la Divina Commedia": "Dante Alighieri", "I promessi sposi": "Alessandro Manzoni",
    "il Decameron": "Giovanni Boccaccio", "il Canzoniere": "Francesco Petrarca",
    "L'infinito": "Giacomo Leopardi", "Il principe": "Niccolò Machiavelli",
    "Il nome della rosa": "Umberto Eco", "Il Gattopardo": "Giuseppe Tomasi di Lampedusa",
    "Le avventure di Pinocchio": "Carlo Collodi", "Il fu Mattia Pascal": "Luigi Pirandello",
    "La coscienza di Zeno": "Italo Svevo", "Se questo è un uomo": "Primo Levi",
    "Le città invisibili": "Italo Calvino", "Orlando furioso": "Ludovico Ariosto",
    "Gerusalemme liberata": "Torquato Tasso", "Amleto": "William Shakespeare",
    "Don Chisciotte": "Miguel de Cervantes", "Guerra e pace": "Lev Tolstoj",
    "I miserabili": "Victor Hugo", "1984": "George Orwell",
}
FACTS = [
    ("Qual è il pianeta più grande del sistema solare?", "Giove", ["Marte", "Saturno", "Venere", "Terra", "Nettuno"]),
    ("Qual è il pianeta più vicino al Sole?", "Mercurio", ["Venere", "Marte", "Terra", "Giove"]),
    ("Quante zampe ha un ragno?", "8", ["6", "10", "4", "12"]),
    ("Quante zampe ha un insetto?", "6", ["8", "4", "10", "12"]),
    ("Qual è l'oceano più grande?", "Pacifico", ["Atlantico", "Indiano", "Artico", "Antartico"]),
    ("Qual è il fiume più lungo d'Italia?", "Po", ["Tevere", "Arno", "Adige", "Piave"]),
    ("Qual è la montagna più alta d'Europa occidentale?", "Monte Bianco", ["Cervino", "Monte Rosa", "Etna", "Gran Sasso"]),
    ("Qual è il vulcano attivo più alto d'Europa?", "Etna", ["Vesuvio", "Stromboli", "Vulcano", "Monte Amiata"]),
    ("A quale temperatura bolle l'acqua al livello del mare?", "100 °C", ["90 °C", "80 °C", "120 °C", "50 °C"]),
    ("Qual è la formula chimica dell'acqua?", "H2O", ["CO2", "O2", "NaCl", "H2O2"]),
    ("Quale gas assorbono le piante durante la fotosintesi?", "Anidride carbonica", ["Ossigeno", "Azoto", "Elio", "Metano"]),
    ("Quanti giorni ha un anno bisestile?", "366", ["365", "364", "360", "367"]),
    ("Quanti minuti ci sono in un'ora?", "60", ["100", "30", "90", "45"]),
    ("Qual è l'organo che pompa il sangue?", "Cuore", ["Fegato", "Polmone", "Rene", "Stomaco"]),
    ("Quante ossa ha circa il corpo umano adulto?", "206", ["106", "306", "150", "256"]),
    ("In che anno è caduto il muro di Berlino?", "1989", ["1991", "1975", "1961", "2001"]),
    ("In che anno è stata proclamata la Repubblica Italiana?", "1946", ["1948", "1861", "1918", "1956"]),
    ("In che anno è avvenuta l'unità d'Italia?", "1861", ["1870", "1848", "1915", "1946"]),
    ("Chi ha dipinto la Gioconda?", "Leonardo da Vinci", ["Michelangelo", "Raffaello", "Caravaggio", "Giotto"]),
    ("Chi ha affrescato la volta della Cappella Sistina?", "Michelangelo", ["Leonardo da Vinci", "Raffaello", "Botticelli", "Tiziano"]),
    ("Chi ha formulato la teoria della relatività?", "Albert Einstein", ["Isaac Newton", "Galileo Galilei", "Niels Bohr", "Marie Curie"]),
    ("Chi ha composto Il barbiere di Siviglia?", "Gioachino Rossini", ["Giuseppe Verdi", "Giacomo Puccini", "Vincenzo Bellini", "Antonio Vivaldi"]),
    ("Chi ha composto Le quattro stagioni?", "Antonio Vivaldi", ["Giuseppe Verdi", "Johann Sebastian Bach", "Wolfgang Amadeus Mozart", "Gioachino Rossini"]),
    ("Qual è la valuta ufficiale dell'Italia?", "Euro", ["Lira", "Dollaro", "Franco", "Sterlina"]),
    ("Quanti sono i continenti tradizionalmente riconosciuti in Italia?", "5", ["7", "6", "4", "3"]),
    ("Qual è il colore che si ottiene mescolando blu e giallo?", "Verde", ["Viola", "Arancione", "Marrone", "Rosa"]),
    ("Quale animale è il più grande mammifero vivente?", "Balenottera azzurra", ["Elefante africano", "Squalo balena", "Giraffa", "Orca"]),
    ("Quale strumento misura la pressione atmosferica?", "Barometro", ["Termometro", "Igrometro", "Anemometro", "Sismografo"]),
    ("Qual è la lingua ufficiale del Brasile?", "Portoghese", ["Spagnolo", "Inglese", "Francese", "Italiano"]),
    ("Quanti lati ha un esagono?", "6", ["5", "7", "8", "4"]),
    ("Qual è la radice quadrata di 81?", "9", ["8", "7", "18", "27"]),
    ("Quale organo del corpo umano produce l'insulina?", "Pancreas", ["Fegato", "Milza", "Tiroide", "Rene"]),
]


def pick_distractors(rng: random.Random, correct: str, pool: list[str], k: int) -> list[str]:
    return rng.sample([p for p in dict.fromkeys(pool) if p != correct], k)


def from_mapping(rng, mapping, template):
    items = []
    values = list(mapping.values())
    for key, value in mapping.items():
        items.append((template.format(key), value, pick_distractors(rng, value, values, 4)))
    return items


def arithmetic(rng, n):
    items = []
    for _ in range(n):
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        op = rng.choice(["+", "×", "-"])
        res = {"+": a + b, "×": a * b, "-": a - b}[op]
        pool = {res + d for d in (-10, -2, -1, 1, 2, 10)} | {res * 2}
        pool.discard(res)
        items.append((f"Quanto fa {a} {op} {b}?", str(res), [str(x) for x in rng.sample(sorted(pool), 4)]))
    return items


def build(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    raw = []
    raw += [("capitali", *x) for x in from_mapping(rng, CAPITALS, "Qual è la capitale di {}?")]
    raw += [("regioni", *x) for x in from_mapping(rng, REGIONS, "Qual è il capoluogo della regione {}?")]
    raw += [("chimica", *x) for x in from_mapping(rng, ELEMENTS, "Qual è il simbolo chimico dell'elemento {}?")]
    raw += [("contrari", *x) for x in from_mapping(rng, OPPOSITES, "Qual è il contrario di \"{}\"?")]
    raw += [("letteratura", *x) for x in from_mapping(rng, WORKS, "Chi è l'autore di {}?")]
    raw += [("cultura generale", q, a, rng.sample(d, 4)) for q, a, d in FACTS]
    raw += [("aritmetica", *x) for x in arithmetic(rng, N_QUESTIONS - len(raw))]
    rng.shuffle(raw)

    out = []
    for i, (category, question, correct, distractors) in enumerate(raw[:N_QUESTIONS]):
        if i < int(N_QUESTIONS * NONE_RATE):
            options = distractors[:4]
            answer = "E"
        else:
            options = distractors[:3] + [correct]
            rng.shuffle(options)
            answer = "ABCD"[options.index(correct)]
        out.append({"category": category, "question": question, "options": options, "answer": answer})
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    questions = build()
    path = Path(__file__).parent / "data" / "test_questions.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(questions, ensure_ascii=False, indent=1))
    counts = {}
    for q in questions:
        counts[q["answer"]] = counts.get(q["answer"], 0) + 1
    print(f"{len(questions)} domande -> {path}  risposte: {dict(sorted(counts.items()))}")
