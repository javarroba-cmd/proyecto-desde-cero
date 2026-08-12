"""
Cliente para la API interna de Codere de la liga "GG League - 4x5 Min de
juego" (DEPORTES > e-Basket > NBA2K > GG League), descubierta inspeccionando
las peticiones de red reales del sitio.

IMPORTANTE:
- Este endpoint SÍ da cuotas pre-partido (isLive=False) con hasta 60-80
  minutos de antelación real, además de partidos ya en curso (isLive=True)
  en la misma respuesta - a diferencia del endpoint de "Directo" que
  usábamos antes, que solo daba partidos ya empezados.
- Codere NO expone histórico de cuotas PASADAS (de partidos ya terminados
  hace tiempo). Para construir un dataset de cuotas hay que capturar
  snapshots repetidos a lo largo del tiempo (ver poll_odds_live.py).
- Sé respetuoso con la frecuencia de peticiones.
"""
import re
import requests
import time

# Endpoint PRE-PARTIDO descubierto inspeccionando la web: DEPORTES > e-Basket >
# NBA2K > "GG League - 4x5 Min de juego". A diferencia del endpoint de
# "Directo" (que solo daba partidos ya empezados), este da tanto partidos
# pre-partido (isLive=false, con hasta 60-80 min de antelación real) como
# partidos ya en curso (isLive=true) en la misma respuesta.
#
# AVISO: GG_LEAGUE_PARENT_ID es el NodeId de la liga "GG League - 4x5 Min de
# juego" en el árbol de navegación de Codere. Codere podría cambiarlo en
# algún momento (nueva temporada, reestructuración). Si un día este ID deja
# de devolver partidos, hay que volver a inspeccionar la web:
#   DEPORTES > e-Basket > NBA2K > GG League - 4x5 Min de juego
# y capturar de nuevo la petición de red GetEvents?parentId=...
GG_LEAGUE_PARENT_ID = "8523428481"

EVENTS_URL = "https://m.apuestas.codere.es/NavigationService/Event/GetEvents"
EVENTS_PARAMS = {"parentId": GG_LEAGUE_PARENT_ID, "gameTypes": "97"}  # 97 = ganador (1x2)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# Ej: "Los Angeles Lakers (BRAZEN)" -> ("Los Angeles Lakers", "BRAZEN")
PARTICIPANT_RE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
CODERE_DATE_RE = re.compile(r"/Date\((\d+)\)/")


def split_team_player(full_name: str) -> tuple[str, str]:
    if not full_name:
        return "", ""
    match = PARTICIPANT_RE.match(full_name.strip())
    if not match:
        return full_name.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def parse_codere_date_ms(value: str) -> int | None:
    """Devuelve los milisegundos Unix del campo de fecha .NET de Codere."""
    if not value:
        return None
    match = CODERE_DATE_RE.match(value)
    return int(match.group(1)) if match else None


def get_gg_league_events() -> list[dict]:
    """Devuelve TODOS los partidos de GG League visibles ahora mismo, tanto
    pre-partido (isLive=False) como ya en curso (isLive=True)."""
    resp = requests.get(EVENTS_URL, headers=HEADERS, params=EVENTS_PARAMS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_moneyline(event: dict) -> list[dict] | None:
    """Extrae los dos resultados del mercado 'Ganador del Partido' (GameTypeId 97)."""
    for game in event.get("Games", []):
        results = game.get("Results", [])
        if results and results[0].get("GameTypeId") == 97:
            return results
    return None


# GameTypeId -> nombre de mercado (deducido por inspección)
GAME_TYPE_NAMES = {
    97: "moneyline_1x2",
    259: "handicap",
    393: "total_puntos",
}


def flatten_event_odds(event: dict, snapshot_time: str) -> list[dict]:
    """Convierte un evento crudo de la API (formato del nuevo endpoint
    GetEvents, válido tanto para pre-partido como en vivo) en filas planas,
    una por resultado de mercado."""
    rows = []
    live = event.get("liveData") or {}  # solo presente si isLive=True
    full_home = event.get("ParticipantHome", "")
    full_away = event.get("ParticipantAway", "")
    team_home, player_home = split_team_player(full_home)
    team_away, player_away = split_team_player(full_away)
    base = {
        "snapshot_time": snapshot_time,
        "node_id": event.get("NodeId"),
        "league_name": event.get("LeagueName"),
        "is_live": event.get("isLive"),
        "team_home": team_home, "player_home": player_home,
        "team_away": team_away, "player_away": player_away,
        "participant_home": full_home,
        "participant_away": full_away,
        "period_name": live.get("PeriodName"),
        "result_home": live.get("ResultHome"),
        "result_away": live.get("ResultAway"),
        "start_date_formatted": event.get("StartDateFormatted"),
    }
    for game in event.get("Games", []):
        for result in game.get("Results", []):
            row = dict(base)
            row.update({
                "game_type_id": result.get("GameTypeId"),
                "game_type_name": GAME_TYPE_NAMES.get(result.get("GameTypeId"), "desconocido"),
                "market_line": result.get("GameSpecialOddsValue"),
                "outcome_name": result.get("Name"),
                "odd": result.get("Odd"),
                "is_locked": result.get("Locked"),
            })
            rows.append(row)
    return rows
